from typing import Any
from astrapy.data import *
from astrapy.geom import *
from astrapy.kernel import (copy_to_symbol, cuda_float4, Kernel)


class RealtimeBackProjection(Kernel):
    # last dim is not number of threads, but volume slab thickness
    VOXELS_PER_BLOCK = (16, 32, 6)
    LIMIT_PROJS_PER_BLOCK = 75

    def __init__(self,
                 voxels_per_block: tuple = None,
                 limit_projs_per_block: int = None):
        super().__init__('realtime.cu', package='jitlearn.cuda')
        self._vox_block = (voxels_per_block if voxels_per_block is not None
                           else self.VOXELS_PER_BLOCK)
        self._limit_projs_per_block = (
            limit_projs_per_block if limit_projs_per_block is not None
            else self.LIMIT_PROJS_PER_BLOCK)

    def compile(self, nr_projs, nr_problems) -> cp.RawModule:
        return self._compile(
            names=('cone_bp',),
            template_kwargs={'nr_vxls_block_x': self._vox_block[0],
                             'nr_vxls_block_y': self._vox_block[1],
                             'nr_vxls_block_z': self._vox_block[2],
                             'nr_projs_block': self._limit_projs_per_block,
                             'nr_projs': nr_projs,
                             'nr_problems': nr_problems })

    def __call__(self,
                 projections_textures: Any,
                 params: cp.ndarray,
                 volumes: cp.ndarray,
                 volume_extent_min: Sequence,
                 volume_extent_max: Sequence):
        """Backprojection with conebeam geometry for multiple volumes.

        All volumes must be equal size, have equal extents, and equal number
        or projection angles.

        `params` must be a `len(volumes) * nr_angles` array containing all the
        parameters for each volume sequentially.
        `projection_textures` must contain all texture references.
         """
        if isinstance(volumes, cp.ndarray):
            if volumes.dtype not in self.SUPPORTED_DTYPES:
                raise NotImplementedError(
                    f"Currently there is only support for "
                    f"dtype={self.SUPPORTED_DTYPES}.")
        else:
            raise TypeError("`volume` must be a CuPy ndarray.")
        assert volumes.flags.c_contiguous is True, (
            f"`{self.__class__.__name__}` is not tested without "
            f"C-contiguous data.")
        for volume in volumes:
            if not has_isotropic_voxels(volume.shape, volume_extent_min,
                                        volume_extent_max):
                raise NotImplementedError(
                    f"`{self.__class__.__name__}` is not tested with anisotropic "
                    f"voxels yet.")

        assert volumes.ndim == 4
        assert params.ndim == 2
        nr_problems = len(volumes)
        nr_angles = len(params) // nr_problems

        projections = cp.array([p.ptr for p in projections_textures])
        vox_volume = voxel_volume(
            volume.shape, volume_extent_min, volume_extent_max)
        module = self.compile(nr_angles, nr_problems)
        cone_bp = module.get_function("cone_bp")
        copy_to_symbol(module, 'params', params.flatten())

        blocks = np.ceil(np.asarray(volume.shape) / self._vox_block).astype(
            np.int32)
        for start in range(0, nr_angles, self._limit_projs_per_block):
            cone_bp((blocks[0] * blocks[1], blocks[2]),  # grid
                    (self._vox_block[0], self._vox_block[1]),  # threads
                    (projections,
                     cp.array([v.data.ptr for v in volumes]),
                     start,
                     *volume.shape,
                     cp.float32(vox_volume)))

    @staticmethod
    def geoms2params(
        geometries: GeometrySequence,
        volume_voxel_size,
        volume_extent_min,
        volume_extent_max,
        volume_rotation=(0., 0., 0.)):
        """Precomputed kernel parameters

        We need three things in the kernel:
         - projected coordinates of pixels on the detector:
          u: || (x-s) v (s-d) || / || u v (s-x) ||
          v: -|| u (x-s) (s-d) || / || u v (s-x) ||
         - ray density weighting factor for the adjoint
          || u v (s-d) ||^2 / ( |cross(u,v)| * || u v (s-x) ||^2 )
         - FDK weighting factor
          ( || u v s || / || u v (s-x) || ) ^ 2

        Since u and v are ratios with the same denominator, we have
        a degree of freedom to scale the denominator. We use that to make
        the square of the denominator equal to the relevant weighting factor.

        For FDK weighting:
            goal: 1/fDen^2 = || u v (s-d) ||^2 / ( |cross(u,v)| * || u v (s-x) ||^2 )
            fDen = ( sqrt(|cross(u,v)|) * || u v (s-x) || ) / || u v (s-d) ||
            i.e. scale = sqrt(|cross(u,v)|) * / || u v (s-d) ||
        Otherwise:
            goal: 1/fDen = || u v s || / || u v (s-x) ||
            fDen = || u v (s-x) || / || u v s ||
            i.e., scale = 1 / || u v s ||
        """
        if isinstance(geometries, list):
            geometries = GeometrySequence.fromList(geometries)
        else:
            geometries = copy.deepcopy(geometries)

        xp = geometries.xp

        normalize_geoms_(geometries, volume_extent_min, volume_extent_max,
                         volume_voxel_size, volume_rotation)

        u = geometries.u * geometries.detector.pixel_width[..., xp.newaxis]
        v = geometries.v * geometries.detector.pixel_height[..., xp.newaxis]
        s = geometries.tube_position
        d = geometries.detector_extent_min

        # NB(ASTRA): for cross(u,v) we invert the volume scaling (for the voxel
        # size normalization) to get the proper dimensions for
        # the scaling of the adjoint
        cr = xp.cross(u, v)  # maintain f32
        cr *= xp.array([volume_voxel_size[1] * volume_voxel_size[2],
                        volume_voxel_size[0] * volume_voxel_size[2],
                        volume_voxel_size[0] * volume_voxel_size[1]])
        scale = (xp.sqrt(xp.linalg.norm(cr, axis=1)) /
                 xp.linalg.det(xp.asarray((u, v, d - s)).swapaxes(0, 1)))

        # TODO(Adriaan): it looks like my preweighting is different to ASTRA's
        #   and I always require voxel-volumetric scaling instead of below
        # if fdk_weighting:
        #     scale = 1. / np.linalg.det([u, v, s])

        _det3x = lambda b, c: b[:, 1] * c[:, 2] - b[:, 2] * c[:, 1]
        _det3y = lambda b, c: b[:, 0] * c[:, 2] - b[:, 2] * c[:, 0]
        _det3z = lambda b, c: b[:, 0] * c[:, 1] - b[:, 1] * c[:, 0]

        s_min_d = s - d
        numerator_u = cuda_float4(
            w=scale * xp.linalg.det(xp.asarray((s, v, d)).swapaxes(0, 1)),
            x=scale * _det3x(v, s_min_d),
            y=-scale * _det3y(v, s_min_d),
            z=scale * _det3z(v, s_min_d))
        numerator_v = cuda_float4(
            w=-scale * xp.linalg.det(xp.asarray((s, u, d)).swapaxes(0, 1)),
            x=-scale * _det3x(u, s_min_d),
            y=scale * _det3y(u, s_min_d),
            z=-scale * _det3z(u, s_min_d))
        denominator = cuda_float4(
            w=scale * xp.linalg.det(xp.asarray((u, v, s)).swapaxes(0, 1)),
            x=-scale * _det3x(u, v),
            y=scale * _det3y(u, v),
            z=-scale * _det3z(u, v))

        # if fdk_weighting:
        #     assert xp.allclose(denominator.w, 1.)

        return cp.asarray(
            numerator_u.to_list() +
            numerator_v.to_list() +
            denominator.to_list()).T


    # @staticmethod
    # def geoms2params(
    #     geometries: GeometrySequence,
    #     vox_size,
    #     volume_extent_min,
    #     volume_extent_max,
    #     volume_rotation=(0., 0., 0.)):
    #     """Precomputed kernel parameters
    #
    #     We need three things in the kernel:
    #      - projected coordinates of pixels on the detector:
    #       u: || (x-s) v (s-d) || / || u v (s-x) ||
    #       v: -|| u (x-s) (s-d) || / || u v (s-x) ||
    #      - ray density weighting factor for the adjoint
    #       || u v (s-d) ||^2 / ( |cross(u,v)| * || u v (s-x) ||^2 )
    #      - FDK weighting factor
    #       ( || u v s || / || u v (s-x) || ) ^ 2
    #
    #     Since u and v are ratios with the same denominator, we have
    #     a degree of freedom to scale the denominator. We use that to make
    #     the square of the denominator equal to the relevant weighting factor.
    #
    #     For FDK weighting:
    #         goal: 1/fDen^2 = || u v (s-d) ||^2 / ( |cross(u,v)| * || u v (s-x) ||^2 )
    #         fDen = ( sqrt(|cross(u,v)|) * || u v (s-x) || ) / || u v (s-d) ||
    #         i.e. scale = sqrt(|cross(u,v)|) * / || u v (s-d) ||
    #     Otherwise:
    #         goal: 1/fDen = || u v s || / || u v (s-x) ||
    #         fDen = || u v (s-x) || / || u v s ||
    #         i.e., scale = 1 / || u v s ||
    #     """
    #     # from tqdm import tqdm
    #     # import itertools
    #     # for _ in tqdm(itertools.count()):
    #     if isinstance(geometries, list):
    #         geometries = GeometrySequence.fromList(geometries)
    #     else:
    #         geometries = copy.deepcopy(geometries)
    #
    #     xp = geometries.XP
    #     normalize_geoms_(geometries, volume_extent_min, volume_extent_max,
    #                      vox_size, volume_rotation)
    #     u = geometries.u * geometries.detector.pixel_width[..., xp.newaxis]
    #     v = geometries.v * geometries.detector.pixel_height[..., xp.newaxis]
    #     s = geometries.tube_position
    #     d = geometries.detector_extent_min
    #
    #     # NB(ASTRA): for cross(u,v) we invert the volume scaling (for the voxel
    #     # size normalization) to get the proper dimensions for
    #     # the scaling of the adjoint
    #     cr = xp.cross(u, v)  # maintain f32
    #     assert vox_size[0] == vox_size[1] == vox_size[2]
    #     cr *= vox_size[0]
    #     scale = xp.sqrt(xp.linalg.norm(cr, axis=1)) / _det(xp, u, v, d - s)
    #
    #     s_min_d = s - d
    #     tmp = xp.empty_like(scale)
    #     w = _det(xp, s, v, d)
    #     x = _det3x(v, s_min_d, tmp)
    #     y = _det3y(v, s_min_d, tmp)
    #     y *= -1
    #     z = _det3z(v, s_min_d, tmp)
    #     w *= scale
    #     x *= scale
    #     y *= scale
    #     z *= scale
    #     numerator_u = cuda_float4(w, x, y, z)
    #     w = _det(xp, s, u, d)
    #     w *= -1
    #     x = _det3x(u, s_min_d, tmp)
    #     x *= -1
    #     y = _det3y(u, s_min_d, tmp)
    #     z = _det3z(u, s_min_d, tmp)
    #     z *= -1
    #     w *= scale
    #     x *= scale
    #     y *= scale
    #     z *= scale
    #     numerator_v = cuda_float4(w, x, y, z)
    #     w = _det(xp, u, v, s)
    #     x = _det3x(u, v, tmp)
    #     x *= -1
    #     y = _det3y(u, v, tmp)
    #     z = _det3z(u, v, tmp)
    #     z *= -1
    #     w *= scale
    #     x *= scale
    #     y *= scale
    #     z *= scale
    #     denominator = cuda_float4(w, x, y, z)
    #     out = cp.asarray(np.asarray(
    #         numerator_u.to_list() +
    #         numerator_v.to_list() +
    #         denominator.to_list()).T)
    #     return out


def _det(xp, a, b, c):
    return xp.linalg.det(xp.asarray((a, b, c)).swapaxes(0, 1))


def _det3x(b, c, tmp):
    out = np.multiply(b[:, 1], c[:, 2])
    np.multiply(b[:, 2], c[:, 1], out=tmp)
    np.subtract(out, tmp, out=out)
    return out


def _det3y(b, c, tmp):
    out = np.multiply(b[:, 0], c[:, 2])
    np.multiply(b[:, 2], c[:, 0], out=tmp)
    np.subtract(out, tmp, out=out)
    return out


def _det3z(b, c, tmp):
    out = np.multiply(b[:, 0], c[:, 1])
    np.multiply(b[:, 1], c[:, 0], out=tmp)
    np.subtract(out, tmp, out=out)
    return out
