from __future__ import annotations

from dataclasses import dataclass, asdict
import numpy as np
from scipy.ndimage import map_coordinates

from .backend import resolve_backend


@dataclass
class SymmetrySettings:
    mode: str = "four-quadrant"
    statistic: str = "mean"
    half_width_px: float = 0.0
    half_height_px: float = 0.0
    interpolation_order: int = 1
    minimum_contributors: int = 1
    compute_backend: str = "auto"
    correlation_sample_target: int = 262144

    def to_dict(self):
        return asdict(self)


@dataclass
class SymmetryResult:
    symmetrized: np.ndarray
    contributors: np.ndarray
    asymmetry_std: np.ndarray
    member_images: np.ndarray
    equator_coordinate_px: np.ndarray
    meridian_coordinate_px: np.ndarray
    center_x: float
    center_y: float
    fiber_angle_deg: float
    settings: SymmetrySettings
    correlations: np.ndarray
    normalized_rms_asymmetry: float

    def summary(self):
        n = self.correlations.shape[0]
        off_diag = ~np.eye(n, dtype=bool)
        finite_corr = self.correlations[off_diag & np.isfinite(self.correlations)]
        return {
            "shape": list(self.symmetrized.shape),
            "center_x_px": float(self.center_x),
            "center_y_px": float(self.center_y),
            "fiber_angle_deg": float(self.fiber_angle_deg),
            "mode": self.settings.mode,
            "statistic": self.settings.statistic,
            "number_of_symmetry_members": int(self.member_images.shape[0]),
            "minimum_contributors": int(self.settings.minimum_contributors),
            "mean_pairwise_member_correlation": float(np.mean(finite_corr)) if finite_corr.size else None,
            "normalized_rms_asymmetry": float(self.normalized_rms_asymmetry) if np.isfinite(self.normalized_rms_asymmetry) else None,
            "ideal_random_noise_snr_gain": float(np.sqrt(self.member_images.shape[0])),
            "compute_backend": resolve_backend(self.settings.compute_backend),
        }


def symmetry_members(mode: str):
    mode = mode.lower().strip()
    if mode == "four-quadrant":
        return [(+1,+1),(-1,+1),(+1,-1),(-1,-1)]
    if mode == "centrosymmetric":
        return [(+1,+1),(-1,-1)]
    if mode == "mirror-meridian":
        return [(+1,+1),(-1,+1)]
    if mode == "mirror-equator":
        return [(+1,+1),(+1,-1)]
    raise ValueError("Unknown symmetry mode.")


def automatic_common_extents(shape, center_x, center_y, fiber_angle_deg):
    h, w = shape
    dx = max(0.0, min(float(center_x), float(w - 1 - center_x)))
    dy = max(0.0, min(float(center_y), float(h - 1 - center_y)))
    if dx <= 0 or dy <= 0:
        raise ValueError("Beam center must lie inside the image with pixels on both sides.")

    theta = np.deg2rad(float(fiber_angle_deg))
    ux, uy = np.cos(theta), np.sin(theta)
    vx, vy = np.cos(theta - np.pi/2), np.sin(theta - np.pi/2)

    def lim(ax, ay):
        vals=[]
        if abs(ax)>1e-12: vals.append(dx/abs(ax))
        if abs(ay)>1e-12: vals.append(dy/abs(ay))
        return min(vals) if vals else 0.0

    e0, m0 = lim(vx,vy), lim(ux,uy)
    xd = e0*abs(vx)+m0*abs(ux)
    yd = e0*abs(vy)+m0*abs(uy)
    scale=1.0
    if xd>0: scale=min(scale,dx/xd)
    if yd>0: scale=min(scale,dy/yd)
    return float(max(1,np.floor(e0*scale))), float(max(1,np.floor(m0*scale)))


def _axes_and_grids(shape, center_x, center_y, angle, settings):
    auto_e, auto_m = automatic_common_extents(shape, center_x, center_y, angle)
    half_e = settings.half_width_px if settings.half_width_px > 0 else auto_e
    half_m = settings.half_height_px if settings.half_height_px > 0 else auto_m
    he, hm = max(1,int(np.floor(half_e))), max(1,int(np.floor(half_m)))
    e_axis=np.arange(-he,he+1,dtype=np.float32)
    m_axis=np.arange(-hm,hm+1,dtype=np.float32)
    eg,mg=np.meshgrid(e_axis,m_axis)
    return e_axis,m_axis,eg,mg


def _aligned_sample_cpu(image, mask, center_x, center_y, angle, eg, mg, order):
    """Resample the detector ONCE into aligned equator/meridian coordinates."""
    theta=np.deg2rad(float(angle))
    ux,uy=np.cos(theta),np.sin(theta)
    vx,vy=np.cos(theta-np.pi/2),np.sin(theta-np.pi/2)
    x=center_x + eg*vx + mg*ux
    y=center_y + eg*vy + mg*uy

    source=np.asarray(image,dtype=np.float32)
    invalid=(~np.isfinite(source)) if mask is None else (np.asarray(mask,dtype=bool)|(~np.isfinite(source)))
    valid=~invalid
    fill=np.float32(np.nanmedian(source[valid]) if np.any(valid) else 0.0)
    values=np.where(invalid,fill,source).astype(np.float32,copy=False)
    aligned=map_coordinates(values,[y,x],order=int(order),mode="constant",cval=np.nan,prefilter=(int(order)>1))
    valid_aligned=map_coordinates(valid.astype(np.uint8),[y,x],order=0,mode="constant",cval=0,prefilter=False)>0
    aligned=np.asarray(aligned,dtype=np.float32)
    aligned[~valid_aligned]=np.nan
    return aligned


def _aligned_sample_gpu(image, mask, center_x, center_y, angle, eg, mg, order):
    import cupy as cp
    from cupyx.scipy.ndimage import map_coordinates as gpu_map_coordinates
    theta=np.deg2rad(float(angle))
    ux,uy=np.cos(theta),np.sin(theta)
    vx,vy=np.cos(theta-np.pi/2),np.sin(theta-np.pi/2)
    eg_g=cp.asarray(eg,dtype=cp.float32); mg_g=cp.asarray(mg,dtype=cp.float32)
    x=cp.float32(center_x)+eg_g*cp.float32(vx)+mg_g*cp.float32(ux)
    y=cp.float32(center_y)+eg_g*cp.float32(vy)+mg_g*cp.float32(uy)
    source=cp.asarray(image,dtype=cp.float32)
    invalid=~cp.isfinite(source)
    if mask is not None: invalid |= cp.asarray(mask,dtype=cp.bool_)
    valid=~invalid
    fill=cp.nanmedian(source[valid]) if bool(cp.any(valid).get()) else cp.float32(0)
    values=cp.where(invalid,fill,source)
    aligned=gpu_map_coordinates(values,cp.stack([y,x]),order=int(order),mode="constant",cval=cp.nan,prefilter=(int(order)>1))
    valid_aligned=gpu_map_coordinates(valid.astype(cp.uint8),cp.stack([y,x]),order=0,mode="constant",cval=0,prefilter=False)>0
    aligned=cp.where(valid_aligned,aligned,cp.nan)
    return cp.asnumpy(aligned).astype(np.float32,copy=False)


def _member_views(aligned, mode):
    views=[]
    for es,ms in symmetry_members(mode):
        v=aligned
        if es<0: v=np.fliplr(v)
        if ms<0: v=np.flipud(v)
        views.append(v)
    return np.stack(views,axis=0).astype(np.float32,copy=False)


def _pairwise_correlations(members,target=262144):
    n=members.shape[0]
    corr=np.full((n,n),np.nan,dtype=np.float32)
    step=max(1,int(np.sqrt(members.shape[1]*members.shape[2]/max(4096,int(target)))))
    sampled=members[:,::step,::step]
    for i in range(n):
        corr[i,i]=1.0
        for j in range(i+1,n):
            a=sampled[i].ravel(); b=sampled[j].ravel()
            valid=np.isfinite(a)&np.isfinite(b)
            if np.count_nonzero(valid)>=20:
                aa=a[valid]; bb=b[valid]
                sa=np.std(aa); sb=np.std(bb)
                value=float(np.corrcoef(aa,bb)[0,1]) if sa>0 and sb>0 else np.nan
            else: value=np.nan
            corr[i,j]=corr[j,i]=value
    return corr


def _combine_members_cpu(members, statistic, minimum):
    finite=np.isfinite(members)
    contributors=np.sum(finite,axis=0,dtype=np.uint8)
    if statistic=="mean":
        sums=np.nansum(members,axis=0,dtype=np.float32)
        sym=np.full(sums.shape,np.nan,dtype=np.float32)
        valid=contributors>0
        sym[valid]=sums[valid]/contributors[valid]
    elif statistic=="median":
        sym=np.full(contributors.shape,np.nan,dtype=np.float32)
        valid=contributors>0
        if np.any(valid): sym[valid]=np.nanmedian(members[:,valid],axis=0).astype(np.float32)
    else:
        raise ValueError("Symmetry statistic must be mean or median.")
    sym[contributors<minimum]=np.nan
    mean=np.nansum(members,axis=0,dtype=np.float32)
    valid=contributors>0
    mean[valid]/=contributors[valid]; mean[~valid]=np.nan
    delta=np.where(finite,members-mean[None,:,:],0.0).astype(np.float32,copy=False)
    ss=np.sum(delta*delta,axis=0,dtype=np.float32)
    std=np.full(mean.shape,np.nan,dtype=np.float32); std[valid]=np.sqrt(ss[valid]/contributors[valid])
    return sym,contributors,std


def _combine_members_gpu(members, statistic, minimum):
    import cupy as cp
    mg=cp.asarray(members,dtype=cp.float32)
    finite=cp.isfinite(mg)
    contributors=cp.sum(finite,axis=0,dtype=cp.uint8)
    if statistic=="mean":
        sums=cp.nansum(mg,axis=0,dtype=cp.float32)
        sym=cp.where(contributors>0,sums/cp.maximum(contributors,1),cp.nan)
    elif statistic=="median":
        sym=cp.nanmedian(mg,axis=0)
    else:
        raise ValueError("Symmetry statistic must be mean or median.")
    sym=cp.where(contributors>=minimum,sym,cp.nan)
    mean=cp.where(contributors>0,cp.nansum(mg,axis=0)/cp.maximum(contributors,1),cp.nan)
    delta=cp.where(finite,mg-mean[None,:,:],0.0)
    std=cp.where(contributors>0,cp.sqrt(cp.sum(delta*delta,axis=0)/cp.maximum(contributors,1)),cp.nan)
    return cp.asnumpy(sym).astype(np.float32),cp.asnumpy(contributors),cp.asnumpy(std).astype(np.float32)


def build_symmetry_average(image,center_x,center_y,fiber_angle_deg,settings=None,mask=None):
    if settings is None: settings=SymmetrySettings()
    image=np.asarray(image,dtype=np.float32)
    if image.ndim!=2: raise ValueError("Symmetry averaging expects a 2-D image.")
    h,w=image.shape
    if not (0<=center_x<=w-1 and 0<=center_y<=h-1): raise ValueError("Beam center lies outside the image.")

    e_axis,m_axis,eg,mg=_axes_and_grids(image.shape,center_x,center_y,fiber_angle_deg,settings)
    backend=resolve_backend(settings.compute_backend)
    if backend=="gpu":
        aligned=_aligned_sample_gpu(image,mask,center_x,center_y,fiber_angle_deg,eg,mg,settings.interpolation_order)
    else:
        aligned=_aligned_sample_cpu(image,mask,center_x,center_y,fiber_angle_deg,eg,mg,settings.interpolation_order)

    members=_member_views(aligned,settings.mode)
    minimum=max(1,int(settings.minimum_contributors))
    if backend=="gpu":
        sym,contributors,std=_combine_members_gpu(members,settings.statistic.lower().strip(),minimum)
    else:
        sym,contributors,std=_combine_members_cpu(members,settings.statistic.lower().strip(),minimum)

    correlations=_pairwise_correlations(members,settings.correlation_sample_target)
    finite=np.isfinite(sym)&np.isfinite(std)
    if np.any(finite):
        signal_rms=float(np.sqrt(np.mean(sym[finite].astype(np.float64)**2)))
        asym_rms=float(np.sqrt(np.mean(std[finite].astype(np.float64)**2)))
        norm_asym=asym_rms/signal_rms if signal_rms>0 else np.nan
    else: norm_asym=np.nan

    return SymmetryResult(
        symmetrized=sym, contributors=contributors, asymmetry_std=std,
        member_images=members, equator_coordinate_px=e_axis,
        meridian_coordinate_px=m_axis, center_x=float(center_x),center_y=float(center_y),
        fiber_angle_deg=float(fiber_angle_deg),settings=settings,correlations=correlations,
        normalized_rms_asymmetry=float(norm_asym),
    )
