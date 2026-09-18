use std::collections::{HashMap, HashSet};

use bevy::{
    app::App,
    asset::{Asset, AssetEvent, AssetId, AssetServer, Assets, Handle, UntypedAssetId},
    ecs::{
        message::MessageReader,
        resource::Resource,
        schedule::IntoScheduleConfigs,
        system::{Res, ResMut, StaticSystemParam, SystemParam, SystemParamItem},
        world::{Mut, World},
    },
    prelude::{Deref, DerefMut, Last},
};
use crossbeam::channel::{Receiver, Sender};

use crate::{
    ray_render_plugin::{RenderSet, TeardownSchedule, on_shutdown},
    render_device::RenderDevice,
};

pub trait VulkanAsset: Asset + Clone + Send + Sync + 'static {
    type ExtractedAsset: Send + Sync + 'static;
    type ExtractParam: SystemParam;
    type PreparedAsset: Send + Sync + 'static;

    fn extract_asset(
        &self,
        param: &mut SystemParamItem<Self::ExtractParam>,
    ) -> Option<Self::ExtractedAsset>;

    /// [`extract_asset`](Self::extract_asset) for assets whose extraction needs to know which
    /// asset it is (a side table keyed by asset path, say).
    fn extract_asset_with_id(
        &self,
        _id: AssetId<Self>,
        param: &mut SystemParamItem<Self::ExtractParam>,
    ) -> Option<Self::ExtractedAsset> {
        self.extract_asset(param)
    }

    fn prepare_asset(
        asset: Self::ExtractedAsset,
        render_device: &RenderDevice,
    ) -> Self::PreparedAsset;

    /// Prepares everything queued since the worker last woke, in order. Override when a batch
    /// can share GPU submissions (BLAS builds do); the default is one-at-a-time.
    fn prepare_batch(
        assets: Vec<Self::ExtractedAsset>,
        render_device: &RenderDevice,
    ) -> Vec<Self::PreparedAsset> {
        assets
            .into_iter()
            .map(|asset| Self::prepare_asset(asset, render_device))
            .collect()
    }
    fn destroy_asset(render_device: &RenderDevice, prepared_asset: &Self::PreparedAsset);
}

/// Upper bound on assets prepared per worker wake-up.
const MAX_PREPARE_BATCH: usize = 1024;

#[derive(Resource)]
pub struct VulkanAssetComms<A: VulkanAsset> {
    send_work: Sender<(AssetId<A>, A::ExtractedAsset)>,
    recv_result: Receiver<(AssetId<A>, A::PreparedAsset)>,
}

impl<A: VulkanAsset> VulkanAssetComms<A> {
    fn new(render_device: RenderDevice) -> Self {
        let (send_work, recv_work) =
            crossbeam::channel::unbounded::<(AssetId<A>, A::ExtractedAsset)>();
        let (send_result, recv_result) = crossbeam::channel::unbounded();

        let ret = Self {
            send_work,
            recv_result,
        };

        std::thread::spawn(move || {
            // Drain whatever piled up while the previous batch was on the GPU, so a scene of
            // thousands of small meshes goes through a few batched submissions instead of a
            // per-mesh submit-and-wait that has to take turns with the frame loop.
            while let Ok(first) = recv_work.recv() {
                let mut ids = vec![first.0];
                let mut assets = vec![first.1];
                while let Ok((id, asset)) = recv_work.try_recv() {
                    ids.push(id);
                    assets.push(asset);
                    if ids.len() >= MAX_PREPARE_BATCH {
                        break;
                    }
                }
                let prepared = A::prepare_batch(assets, &render_device);
                debug_assert_eq!(prepared.len(), ids.len());
                for item in ids.into_iter().zip(prepared) {
                    if send_result.send(item).is_err() {
                        return;
                    }
                }
            }
        });

        ret
    }
}

pub enum VulkanAssetLoadingState<A: VulkanAsset> {
    Loading,
    Loaded(A::PreparedAsset),
}

#[derive(Resource, Deref, DerefMut)]
pub struct VulkanAssets<A: VulkanAsset>(HashMap<AssetId<A>, VulkanAssetLoadingState<A>>);

impl<A: VulkanAsset> VulkanAssets<A> {
    pub fn get(&self, handle: &Handle<A>) -> Option<&A::PreparedAsset> {
        self.get_by_id(handle.id())
    }

    pub fn get_by_id(&self, id: AssetId<A>) -> Option<&A::PreparedAsset> {
        self.0.get(&id).map_or(None, |state| match state {
            VulkanAssetLoadingState::Loading => None,
            VulkanAssetLoadingState::Loaded(asset) => Some(asset),
        })
    }
}

impl<A: VulkanAsset> Default for VulkanAssets<A> {
    fn default() -> Self {
        Self(HashMap::default())
    }
}

/// Ids whose last handle dropped while their build was still in flight: the result is
/// destroyed on arrival instead of being kept for nobody.
#[derive(Resource)]
pub struct VulkanAssetDropped<A: VulkanAsset>(HashSet<AssetId<A>>);

impl<A: VulkanAsset> Default for VulkanAssetDropped<A> {
    fn default() -> Self {
        Self(HashSet::default())
    }
}

/// Assets whose last handle dropped this frame (their GPU side is already on the destroyer);
/// `prepare_instances` releases their SBT hit records.
#[derive(Resource, Default)]
pub struct DroppedAssets(pub Vec<UntypedAssetId>);

/// Prepared assets replaced this frame (a re-prepare after `AssetEvent::Modified`, or a
/// duplicate load). The old GPU objects are already queued on the destroyer, so anything
/// holding their addresses -- TLAS instance rows above all -- must re-resolve THIS frame:
/// `prepare_instances` drains this after every `poll_for_asset`.
#[derive(Resource, Default)]
pub struct ReplacedAssets(pub Vec<UntypedAssetId>);

fn extract_vulkan_asset<A: VulkanAsset>(
    mut asset_events: MessageReader<AssetEvent<A>>,
    assets: Res<Assets<A>>,
    asset_server: Res<AssetServer>,
    render_device: Res<RenderDevice>,
    mut render_assets: ResMut<VulkanAssets<A>>,
    mut dropped: ResMut<VulkanAssetDropped<A>>,
    mut dropped_ids: ResMut<DroppedAssets>,
    comms: Res<VulkanAssetComms<A>>,
    param: StaticSystemParam<A::ExtractParam>,
) {
    let mut param = param.into_inner();
    for event in asset_events.read() {
        match event {
            AssetEvent::Added { id } | AssetEvent::LoadedWithDependencies { id } => {
                log::debug!("VulkanAsset received {event:?}");
                // One build per asset: `Added` and `LoadedWithDependencies` both land here
                // (often in the same frame), and a late `LoadedWithDependencies` must not
                // overwrite an entry that already holds the prepared asset.
                if render_assets.0.contains_key(id) {
                    continue;
                }
                if let Some(asset) = assets.get(*id) {
                    if let Some(extracted) = asset.extract_asset_with_id(*id, &mut param) {
                        render_assets.insert(*id, VulkanAssetLoadingState::Loading);
                        comms.send_work.send((*id, extracted)).unwrap();
                    }
                } else {
                    log::warn!("VulkanAsset could not find asset with id: {:?}", id);
                }
            }
            AssetEvent::Modified { id } => {
                // A re-prepare: the replacement lands through `poll_for_asset`, which
                // reports it in `ReplacedAssets` so dependents re-resolve before the old
                // GPU objects are destroyed.
                log::debug!(
                    "VulkanAsset received AssetEvent::Modified for {:?} ({:?})",
                    id,
                    asset_server.get_path(*id)
                );
                if let Some(asset) = assets.get(*id) {
                    if let Some(extracted) = asset.extract_asset_with_id(*id, &mut param) {
                        comms.send_work.send((*id, extracted)).unwrap();
                    }
                } else {
                    log::warn!("VulkanAsset could not find asset with id: {:?}", id);
                }
            }
            AssetEvent::Unused { id } | AssetEvent::Removed { id } => {
                // No handle left: free the GPU side. Every instance built on it was
                // despawned before its handle could drop, so the TLAS rows are already
                // zero; the destroyer's delay covers the frame in flight.
                match render_assets.0.remove(id) {
                    Some(VulkanAssetLoadingState::Loaded(prepared)) => {
                        log::debug!("VulkanAsset dropped {id:?}");
                        A::destroy_asset(&render_device, &prepared);
                        dropped_ids.0.push(id.untyped());
                    }
                    Some(VulkanAssetLoadingState::Loading) => {
                        dropped.0.insert(*id);
                        dropped_ids.0.push(id.untyped());
                    }
                    None => {}
                }
            }
        }
    }
}

pub fn poll_for_asset<A: VulkanAsset>(
    render_device: Res<RenderDevice>,
    comms: Res<VulkanAssetComms<A>>,
    mut assets: ResMut<VulkanAssets<A>>,
    mut dropped: ResMut<VulkanAssetDropped<A>>,
    mut replaced: ResMut<ReplacedAssets>,
) {
    while let Ok((id, prep)) = comms.recv_result.try_recv() {
        log::debug!("VulkanAsset received prepared asset for id: {:?}", id);
        if dropped.0.remove(&id) {
            log::debug!("VulkanAsset {id:?} dropped while building; destroying the result");
            A::destroy_asset(&render_device, &prep);
            continue;
        }
        if let Some(old) = assets.0.insert(id, VulkanAssetLoadingState::Loaded(prep)) {
            match old {
                VulkanAssetLoadingState::Loading => {}
                VulkanAssetLoadingState::Loaded(old) => {
                    // The destroyer delays this past the in-flight frame, but NOT past the
                    // next one: whoever holds the old addresses (TLAS rows, SBT records)
                    // must pick up the replacement in this frame's prepare.
                    log::debug!("VulkanAsset replaced prepared asset {id:?}; old destroyed");
                    replaced.0.push(id.untyped());
                    A::destroy_asset(&render_device, &old);
                }
            }
        }
    }
}

fn on_shutdown_asset<A: VulkanAsset>(world: &mut World) {
    let comms = world.remove_resource::<VulkanAssetComms<A>>();
    world.resource_scope(|world, mut assets: Mut<VulkanAssets<A>>| {
        let render_device = world.get_resource::<RenderDevice>().unwrap();
        // Let the worker finish what it has (it records into the device and binds
        // pipelines that are about to go); each job it took yields one result, and its
        // result channel closes once it exits.
        if let Some(VulkanAssetComms {
            send_work,
            recv_result,
        }) = comms
        {
            drop(send_work);
            let mut finished = 0usize;
            while let Ok((id, prep)) = recv_result.recv() {
                assets.0.remove(&id);
                A::destroy_asset(&render_device, &prep);
                finished += 1;
            }
            if finished > 0 {
                log::info!("VulkanAsset: waited for {finished} in-flight build(s) at shutdown");
            }
        }
        for (_, prep) in assets.0.drain() {
            match prep {
                VulkanAssetLoadingState::Loading => {
                    log::warn!("VulkanAsset was still loading when shutting down");
                }
                VulkanAssetLoadingState::Loaded(prep) => A::destroy_asset(&render_device, &prep),
            }
        }
    });
}

pub trait VulkanAssetExt {
    fn init_vulkan_asset<A: VulkanAsset>(&mut self);
}

impl VulkanAssetExt for App {
    fn init_vulkan_asset<A: VulkanAsset>(&mut self) {
        let render_device = self.world().resource::<RenderDevice>().clone();
        self.insert_resource(VulkanAssetComms::<A>::new(render_device));
        self.init_resource::<VulkanAssets<A>>();
        self.init_resource::<VulkanAssetDropped<A>>();
        self.init_resource::<ReplacedAssets>();
        self.init_resource::<DroppedAssets>();
        self.add_systems(
            Last,
            (
                extract_vulkan_asset::<A>.in_set(RenderSet::Extract),
                poll_for_asset::<A>.in_set(RenderSet::Prepare),
            ),
        );
        self.add_systems(TeardownSchedule, on_shutdown_asset::<A>.before(on_shutdown));
    }
}
