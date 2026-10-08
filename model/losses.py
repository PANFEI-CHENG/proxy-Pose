import math
import torch
import torch.nn as nn
import torch.nn.functional as F

def SmoothL1Dis(p1, p2, threshold=0.1):
    '''
    p1: b*n*3
    p2: b*n*3
    '''
    diff = torch.abs(p1 - p2)
    less = torch.pow(diff, 2) / (2.0 * threshold)
    higher = diff - threshold / 2.0
    dis = torch.where(diff > threshold, higher, less)
    dis = torch.mean(torch.sum(dis, dim=2))
    return dis

def ChamferDis(p1, p2):
    '''
    p1: b*n1*3
    p2: b*n2*3
    '''
    dis = torch.norm(p1.unsqueeze(2) - p2.unsqueeze(1), dim=3)
    dis1 = torch.min(dis, 2)[0]
    dis2 = torch.min(dis, 1)[0]
    dis = 0.5*dis1.mean(1) + 0.5*dis2.mean(1)
    return dis.mean()

def ChamferDis_wo_Batch(p1, p2):
    """
    Args:
        p1: (n1, 3)
        p2: (n2, 3)
    """
    dis = torch.norm(p1.unsqueeze(1) - p2.unsqueeze(0), dim=2) # (n1, n2)
    dis1 = torch.min(dis, 1)[0] # (n1, )
    dis2 = torch.min(dis, 0)[0] # (n2, )
    dis = 0.5*dis1.mean() + 0.5*dis2.mean()
    return dis

def PoseDis(r1, t1, s1, r2, t2, s2):
    '''
    r1, r2: b*3*3
    t1, t2: b*3
    s1, s2: b*3
    '''
    dis_r = torch.mean(torch.norm(r1 - r2, dim=1))
    dis_t = torch.mean(torch.norm(t1 - t2, dim=1))
    dis_s = torch.mean(torch.norm(s1 - s2, dim=1))

    return dis_r + dis_t + dis_s

def UniChamferDis(p1, p2):
    '''
    p1: b, n1, 3
    p2: b, n2, 3
    '''
    # (b, n1, n2)
    dis = torch.norm(p1.unsqueeze(2) - p2.unsqueeze(1), dim=3)
    dis = torch.min(dis, 2)[0]

    return dis.mean()


class ComPoseProxyLoss(nn.Module):
    def __init__(self, cfg, num_obj=64, axis_order='xyz', sym_ids=(0, 1, 3),
                 axis_length_ratio=0.8, axis_diameter_nocs=0.12, yaw_bins=36,
                 gt_model_points=1024, chamfer_chunk=128, eps=1e-7):
        super().__init__()
        self.cfg = cfg
        self.num_obj = int(num_obj)
        self.axis_order = str(axis_order).lower()
        if len(self.axis_order) != 3 or set(self.axis_order) != {'x', 'y', 'z'}:
            raise ValueError('axis_order must be a permutation of xyz')
        self.sym_ids = tuple(int(v) for v in sym_ids)
        self.axis_ratio = float(axis_length_ratio)
        self.half_width = float(axis_diameter_nocs) / 2.0
        self.yaw_bins = int(yaw_bins)
        if not 0 < axis_diameter_nocs < 1:
            raise ValueError('axis_diameter_nocs must be in (0,1)')
        if self.yaw_bins < 8:
            raise ValueError('yaw_bins must be >= 8')
        self.gt_model_points = int(gt_model_points)
        self.chamfer_chunk = int(chamfer_chunk)
        self.eps = eps
        if not 0 < self.axis_ratio <= 1:
            raise ValueError('axis_length_ratio must be in (0,1]')

    def weight(self, key, default):
        return float(getattr(self.cfg, key, default))

    @staticmethod
    def _huber(a, b, beta=0.05):
        return F.smooth_l1_loss(a, b, beta=beta, reduction='none')

    def _chamfer(self, a, b, symmetric=True):
        """Memory-aware Chamfer using torch.cdist chunks, coordinates in the input units."""
        if a.size(1) == 0 or b.size(1) == 0:
            return a.new_zeros(())
        d_a = []
        d_b = None
        for start in range(0, a.size(1), self.chamfer_chunk):
            part = torch.cdist(a[:, start:start+self.chamfer_chunk], b)
            d_a.append(part.min(dim=2).values)
            if symmetric:
                part_b = part.min(dim=1).values
                d_b = part_b if d_b is None else torch.minimum(d_b, part_b)
        forward = torch.cat(d_a, dim=1).mean()
        return 0.5 * (forward + d_b.mean()) if symmetric else forward

    @staticmethod
    def _sdf_box(points, half_dims):
        """p:[B,P,3], half_dims:[B,3]; exact axis-aligned box SDF."""
        q = points.abs() - half_dims[:, None, :]
        outside = F.relu(q).norm(dim=-1)
        inside = q.amax(dim=-1).clamp(max=0.0)
        return outside + inside

    def _sdf_cylinder_y(self, points, half_len):
        """Finite capped Y cylinder in NOCS units, radius=axis_diameter_nocs/2."""
        radial = torch.linalg.vector_norm(points[..., [0, 2]], dim=-1) - self.half_width
        axial = points[..., 1].abs() - half_len[:, None]
        q = torch.stack([radial, axial], dim=-1)
        return F.relu(q).norm(dim=-1) + q.amax(dim=-1).clamp(max=0.0)

    def _surface_samples(self, axis, half_len):
        """Analytical deterministic surface samples in local NOCS coordinates.

        For cylinder: 12 angular x 5 axial side points + 2x(9 disk points).
        For boxes: four sides x (5 length x 3 cross) + two caps x 3x3.
        """
        b = half_len.size(0)
        device, dtype = half_len.device, half_len.dtype
        h = half_len[:, None]
        w = self.half_width
        if axis == 'y':
            ang = torch.arange(12, dtype=dtype, device=device) * (2*math.pi/12)
            yunit = torch.linspace(-1.0, 1.0, 5, dtype=dtype, device=device)
            x = w * ang.cos().repeat(5)
            z = w * ang.sin().repeat(5)
            y = (h * yunit.repeat_interleave(12)[None, :]).expand(b, -1)
            side = torch.stack([x.expand(b, -1), y, z.expand(b, -1)], dim=-1)
            angles = torch.arange(8, dtype=dtype, device=device) * (2*math.pi/8)
            xc = torch.cat([w*angles.cos(), angles.new_zeros(1)])
            zc = torch.cat([w*angles.sin(), angles.new_zeros(1)])
            cap = torch.stack([xc.expand(b, -1), torch.zeros((b, 9),device=device,dtype=dtype), zc.expand(b,-1)],dim=-1)
            lower = cap.clone(); lower[...,1] = -h
            upper = cap.clone(); upper[...,1] = h
            return torch.cat([side, lower, upper], dim=1)
        # Boxes along x and z, with cross-section sides at +/- w.
        long = torch.linspace(-1, 1, 5, dtype=dtype, device=device)
        cross = torch.linspace(-w, w, 3, dtype=dtype, device=device)
        coord_l = (h * long.repeat_interleave(3)[None,:]).expand(b, -1)
        coord_c = cross.repeat(5).expand(b, -1)
        sides = []
        for fix_dim in (1, 2) if axis == 'x' else (0, 1):
            for sign in (-1.0,1.0):
                part = torch.zeros((b,15,3), device=device, dtype=dtype)
                along = 0 if axis == 'x' else 2
                other = next(k for k in range(3) if k not in (along,fix_dim))
                part[..., along] = coord_l
                part[..., other] = coord_c
                part[..., fix_dim] = sign*w
                sides.append(part)
        cg = torch.cartesian_prod(cross, cross)
        caps=[]
        along = 0 if axis == 'x' else 2
        others=[k for k in range(3) if k != along]
        for sign in (-1.0,1.0):
            cap=torch.zeros((b,9,3),device=device,dtype=dtype)
            cap[..., along] = sign*h
            cap[...,others[0]] = cg[:,0]
            cap[...,others[1]] = cg[:,1]
            caps.append(cap)
        return torch.cat(sides+caps,dim=1)

    @staticmethod
    def _yaw_rotate_row(points, theta):
        """Apply one shared yaw to X/Z proxy groups (row-vector rotation convention).

        points: [B,N,3] or [B,A,N,3]; theta: [B] or [B,A].
        """
        cs, sn = theta.cos()[..., None], theta.sin()[..., None]
        x = points[..., 0] * cs - points[..., 2] * sn
        z = points[..., 0] * sn + points[..., 2] * cs
        return torch.stack([x, points[..., 1], z], dim=-1)

    def _axis_losses(self, group, axis, half_len):
        """Point-to-surface SDF + target-surface coverage for an axis group, in NOCS."""
        b = group.size(0)
        if axis == 'y':
            sdf = self._sdf_cylinder_y(group, half_len)
        else:
            half_dims = group.new_full((b, 3), self.half_width)
            half_dims[:, 'xyz'.index(axis)] = half_len
            sdf = self._sdf_box(group, half_dims)
        surface = sdf.abs().mean(dim=1)
        target = self._surface_samples(axis, half_len)
        coverage = torch.cdist(target, group).min(dim=-1).values.mean(dim=-1)
        return surface, coverage

    def _optimal_symmetric_yaw(self, x_group, z_group, hx, hz):
        """Jointly fit X and Z boxes under ONE free yaw; chooses a grid orientation.

        This is not independent pointwise yaw minimization: X/Z remain a rigid
        perpendicular skeleton. Selection is non-differentiable, but the loss
        at the selected yaw remains differentiable w.r.t. predicted points.
        """
        b = x_group.size(0)
        a = self.yaw_bins
        with torch.no_grad():
            angles = torch.arange(a, dtype=x_group.dtype, device=x_group.device) * (2.0 * math.pi / a)
            theta = angles.view(1, a).expand(b, a)
            gx = self._yaw_rotate_row(x_group[:, None].expand(-1, a, -1, -1), theta)
            gz = self._yaw_rotate_row(z_group[:, None].expand(-1, a, -1, -1), theta)
            ax, cx = self._axis_losses(gx.reshape(b * a, -1, 3), 'x', hx.repeat_interleave(a))
            az, cz = self._axis_losses(gz.reshape(b * a, -1, 3), 'z', hz.repeat_interleave(a))
            # Both surface and coverage must decide yaw: surface-only admits collapse.
            score = ax.reshape(b, a) + az.reshape(b, a) + cx.reshape(b, a) + cz.reshape(b, a)
            best = score.argmin(dim=1)
            return angles[best]

    def _proxy_geom(self, proxy_nocs, size_gt, symmetric):
        """All three axes supervised for ALL classes; symmetric X/Z have free yaw.

        proxy_nocs: [B,P,3], canonical coordinates normalized by ||size_gt||.
        size_gt: [B,3], physical extents in metres; used only as ratios.
        X/Z: finite boxes with dimensionless square cross-section.
        Y: finite cylinder with dimensionless diameter axis_diameter_nocs.
        Full axis lengths: axis_length_ratio * (size_gt[i]/||size_gt||).
        The same X/Z yaw is optimized jointly for symmetric instances.
        """
        b, p, _ = proxy_nocs.shape
        if p % 3:
            raise ValueError(f'Proxy count {p} must split into 3 axis groups')
        g = dict(zip(self.axis_order, proxy_nocs.split(p // 3, dim=1)))
        extent_norm = size_gt / size_gt.norm(dim=-1, keepdim=True).clamp_min(self.eps)
        half_lens = {axis: (extent_norm[:, 'xyz'.index(axis)] * self.axis_ratio * 0.5).clamp_min(self.eps)
                     for axis in 'xyz'}
        # A genuinely Y-axis symmetric object has one shared radial size, not
        # two independently labeled x/z lengths (which may contain label noise).
        radial_half_len = (0.25 * self.axis_ratio * (extent_norm[:, 0] + extent_norm[:, 2])).clamp_min(self.eps)
        half_lens['x'] = torch.where(symmetric, radial_half_len, half_lens['x'])
        half_lens['z'] = torch.where(symmetric, radial_half_len, half_lens['z'])

        # For asymmetric samples theta=0 (canonical XYZ must match GT pose).
        # For symmetric samples theta is a shared free yaw for the X/Z skeleton.
        if bool(symmetric.any()):
            theta_best = self._optimal_symmetric_yaw(g['x'], g['z'], half_lens['x'], half_lens['z'])
            theta = torch.where(symmetric, theta_best, theta_best.new_zeros(()))
            g['x'] = self._yaw_rotate_row(g['x'], theta)
            g['z'] = self._yaw_rotate_row(g['z'], theta)

        surfaces, coverages = [], []
        for axis in 'xyz':
            s, c = self._axis_losses(g[axis], axis, half_lens[axis])
            surfaces.append(s)
            coverages.append(c)
        return torch.stack(surfaces, dim=-1).mean(), torch.stack(coverages, dim=-1).mean()

    def _nocs_terms(self, nocs, xyz, rotation, translation, size, symmetric):
        """For symmetry supervise radius/Y on EVERY point, including X/Z proxies.

        Pairwise normalized distances are yaw-invariant and include all pairs.
        Object and proxy canonical correspondences are kept geometrically linked.
        """
        scale = size.norm(dim=-1).clamp_min(self.eps)
        canonical = torch.bmm(xyz.detach() - translation[:, None], rotation) / scale[:, None, None]
        full_loss = self._huber(nocs, canonical).sum(-1)
        nocs_ry = torch.stack([nocs[..., [0,2]].norm(dim=-1), nocs[..., 1]], dim=-1)
        gt_ry = torch.stack([canonical[..., [0,2]].norm(dim=-1), canonical[..., 1]], dim=-1)
        sym_loss = self._huber(nocs_ry, gt_ry).sum(-1)
        nocs_loss = torch.where(symmetric[:,None], sym_loss, full_loss).mean()

        # ComPose-inspired geometry consistency, including *all* Object-Proxy pairs.
        d_camera = torch.cdist(xyz.detach(), xyz.detach()) / scale[:, None, None]
        d_nocs = torch.cdist(nocs, nocs)
        b, n, _ = xyz.shape
        off_diag = ~torch.eye(n, device=xyz.device, dtype=torch.bool)
        geo = (d_camera - d_nocs).square().masked_select(off_diag[None].expand(b,-1,-1)).mean()
        return nocs_loss, geo

    @staticmethod
    def _pose_component(rp, tp, sp, rg, tg, sg, symmetric):
        # For axial symmetry, only the canonical Y axis is observable.
        full = torch.linalg.vector_norm(rp-rg,dim=(1,2))
        axis = torch.linalg.vector_norm(rp[:,:,1]-rg[:,:,1],dim=-1)
        rot = torch.where(symmetric,axis,full).mean()
        trans = torch.linalg.vector_norm(tp-tg,dim=-1).mean()
        dims = torch.linalg.vector_norm(sp-sg,dim=-1).mean()
        return rot+trans+dims, rot

    def forward(self, endpoints):
        """Preserves existing endpoints from Net.forward and train Solver."""
        x = endpoints['pred_kpt_3d'].float()
        nocs = endpoints['pred_kpt_nocs'].float()
        recon = endpoints['recon_model'].float()
        delta = endpoints['recon_delta'].float()
        R = endpoints['rotation_label'].float()
        t = endpoints['translation_label'].float()
        s = endpoints['size_label'].float()
        cls = endpoints['category_label'].reshape(-1).long()
        sym = torch.zeros_like(cls,dtype=torch.bool)
        for k in self.sym_ids:
            sym |= cls == k
        b, n, _ = x.shape
        if not self.num_obj < n or nocs.shape != x.shape:
            raise ValueError('Expected joint Object+Proxy predictions and matching NOCS')
        nproxy=n-self.num_obj
        if nproxy%3:
            raise ValueError('Expected 3 equal-size proxy groups')
        obj=x[:,:self.num_obj]
        proxy=x[:,self.num_obj:]
        scale=torch.linalg.vector_norm(s,dim=-1).clamp_min(self.eps)
        local_proxy=torch.bmm(proxy-t[:,None],R) / scale[:,None,None]  # dimensionless GT NOCS frame

        with torch.amp.autocast(device_type=x.device.type,enabled=False):
            proxy_surface,proxy_coverage=self._proxy_geom(local_proxy,s,sym)
            proxy_nocs_surface,proxy_nocs_coverage=self._proxy_geom(nocs[:,self.num_obj:],s,sym)
            nocs_loss,geo=self._nocs_terms(nocs,x,R,t,s,sym)
            pose,rot_error=self._pose_component(endpoints['pred_rotation'].float(),
                endpoints['pred_translation'].float(),endpoints['pred_size'].float(),R,t,s,sym)
            if all(k in endpoints for k in ('pred_init_rotation','pred_init_translation','pred_init_scale')):
                init_s=torch.linalg.vector_norm(endpoints['pred_init_rotation'].float() - R,dim=(1,2))
                init_axis=torch.linalg.vector_norm(endpoints['pred_init_rotation'].float()[:,:,1]-R[:,:,1],dim=-1)
                init_rot=torch.where(sym,init_axis,init_s).mean()
                init_t=torch.linalg.vector_norm(endpoints['pred_init_translation'].float()-t,dim=-1).mean()
                init_scale=F.smooth_l1_loss(endpoints['pred_init_scale'].float(),scale)
                pose_init=init_rot+init_t+init_scale
            else:
                pose_init=x.new_zeros(())
            if 'model' in endpoints:
                # Dataset model is dimensionless NOCS, transform GT CAD to metric camera frame.
                model=endpoints['model'].float()
                if model.ndim != 3 or model.shape[-1] != 3:
                    raise ValueError(f'Expected CAD model [B,M,3], got {model.shape}')
                if model.size(1)>self.gt_model_points:
                    ids=torch.linspace(0,model.size(1)-1,self.gt_model_points,device=x.device).long()
                    model=model.index_select(1,ids)
                model_cam=torch.bmm(model * scale[:,None,None],R.transpose(1,2))+t[:,None]
                model_cam=model_cam.detach()
                recon_cd=self._chamfer(recon,model_cam,symmetric=True)
                object_cd=self._chamfer(obj,model_cam,symmetric=True)
            else:
                # HouseCat6D lacks CAD model in its current loader: partial coverage only.
                partial=endpoints['pts'].float()
                recon_cd=self._chamfer(partial,recon,symmetric=False)
                object_cd=obj.new_zeros(())
            # Object-only diversity: proxy points should NOT be repelled as surface keypoints.
            threshold=float(getattr(self.cfg,'th',0.01))
            distance=torch.cdist(obj,obj)
            off_diag=1-torch.eye(self.num_obj,device=obj.device,dtype=obj.dtype)
            diversity=((threshold-distance).clamp_min(0)*off_diag).sum((1,2)) / (
                self.num_obj*(self.num_obj-1)*max(threshold,self.eps))
            diversity=diversity.mean()
            delta_loss=delta.norm(dim=-1).mean()

            terms={
                'loss_pose':self.weight('pose',0.3)*pose,
                'loss_pose_init':self.weight('pose_init',0.1)*pose_init,
                'loss_recon':self.weight('recon',15.0)*recon_cd,
                'loss_obj_cd':self.weight('obj_cd',2.0)*object_cd,
                'loss_proxy_surface':self.weight('proxy_surface',2.0)*proxy_surface,
                'loss_proxy_coverage':self.weight('proxy_coverage',1.0)*proxy_coverage,
                'loss_proxy_nocs_surface':self.weight('proxy_nocs_surface',1.0)*proxy_nocs_surface,
                'loss_proxy_nocs_coverage':self.weight('proxy_nocs_coverage',0.5)*proxy_nocs_coverage,
                'loss_nocs':self.weight('nocs',2.0)*nocs_loss,
                'loss_geo':self.weight('geo',1.0)*geo,
                'loss_diversity':self.weight('diversity',1.0)*diversity,
                'loss_delta':self.weight('delta',1.0)*delta_loss,
            }
            terms['loss_all']=sum(terms.values())
            # Extra scalar diagnostics; safe to log through existing Solver.
            terms['loss_rotation_unweighted']=rot_error.detach()
            terms['loss_proxy_surface_unweighted']=proxy_surface.detach()
            terms['loss_proxy_coverage_unweighted']=proxy_coverage.detach()
            terms['loss_proxy_nocs_surface_unweighted']=proxy_nocs_surface.detach()
            return terms