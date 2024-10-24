import torch
from einops import rearrange

# copied from
def pixel_patchify(pixel_values, tubelet_size):
    """
    (B,C,H,W) -> (B,num_patches, c, h, w)
    """
    B,C,H,W = pixel_values.size()
    n_h = H//tubelet_size[1]
    n_w = W//tubelet_size[2]

    pixel_values = rearrange(pixel_values, 'b c (n_h h) (n_w w) -> b c n_h h n_w w', n_h=n_h, n_w=n_w)
    pixel_values = rearrange(pixel_values, 'b c n_h h n_w w -> b (n_h n_w) c h w')

    return pixel_values

def pixel_unpatchify(pixel_values, tubelet_size, image_size):
    """
    (b (n_h n_w) c h w) -> (b c (n_h h) (n_w w))
    """
    B, num_patches, c, h, w = pixel_values.size()
    n_h = image_size //tubelet_size[1]
    n_w = image_size // tubelet_size[2]

    # (b (n_t n_h n_w) t c h w) -> (b (n_t t) c (n_h h) (n_w w))
    pixel_values = rearrange(pixel_values,'b (n_h n_w) c h w -> b c (n_h h) (n_w w)', n_h=n_h, n_w=n_w)

    return pixel_values

def count_weight_for_patch_weighted_mae(config, pixel):
    patchified_pixel = pixel_patchify(config, pixel)
    B, num_patches, t, c, h, w = patchified_pixel.size()
    # 初始化权重张量
    weights = torch.zeros_like(patchified_pixel)

    # 计算权重
    for b in range(B):
        for p in range(num_patches):
            for ti in range(t):
                patch = patchified_pixel[b, p, ti, 0, :, :]  # (h, w)
                # print("块大小：",patch.size())
                
                # 找到正值和负值的索引
                pos_indices = torch.nonzero(patch > 0, as_tuple=False)
                neg_indices = torch.nonzero(patch < 0, as_tuple=False)
                
                if len(pos_indices) > 0 and len(neg_indices) > 0:
                    # 计算正值中心
                    pos_center = pos_indices.float().mean(dim=0)
                    # print("正值中心",pos_center.size(), pos_indices.float().size())
                    # 计算负值中心
                    neg_center = neg_indices.float().mean(dim=0)
                    # 计算权重
                    weight_value = torch.norm(pos_center - neg_center).item()
                    weight_value = weight_value*patch.abs().max()
                    weights[b, p, ti, 0, :, :] = weight_value
    # 把weights reshape回去(B,num_patches,t, c, h, w)->(B,T,C,H,W)
    token_weights = weights.mean(dim = (-1,-2,-3,-4)) # (b, p)
    weights = pixel_unpatchify(config, weights)
    weights = weights/weights.sum()

    return weights, token_weights

def count_weight_for_patch_weighted_mae_cuda(config, pixel):
    patchified_pixel = pixel_patchify(config, pixel)
    B, num_patches, t, c, h, w = patchified_pixel.size()
    # 初始化权重张量
    weights = torch.zeros_like(patchified_pixel)

    # 并行计算权重
    # 统计每个patch(c, h, w)上的正值和负值的索引
    token_weights = weights.mean(dim = (-1,-2,-3,-4)) # (b, p)
    weights = pixel_unpatchify(config, weights)
    weights = weights/weights.sum()

    return weights, token_weights

def count_weight_for_patch_weighted_expmae(config, pixel):
    patchified_pixel = pixel_patchify(config, pixel)
    B, num_patches, t, c, h, w = patchified_pixel.size()
    # 初始化权重张量
    weights = torch.zeros_like(patchified_pixel)

    # 计算权重
    for b in range(B):
        for p in range(num_patches):
            for ti in range(t):
                patch = patchified_pixel[b, p, ti, 0, :, :]  # (h, w)
                # print("块大小：",patch.size())
                
                # 找到正值和负值的索引
                pos_indices = torch.nonzero(patch > 0, as_tuple=False)
                neg_indices = torch.nonzero(patch < 0, as_tuple=False)
                
                if len(pos_indices) > 0 and len(neg_indices) > 0:
                    # 计算正值中心
                    pos_center = pos_indices.float().mean(dim=0)
                    # print("正值中心",pos_center.size(), pos_indices.float().size())
                    # 计算负值中心
                    neg_center = neg_indices.float().mean(dim=0)
                    # 计算权重
                    weight_value = torch.norm(pos_center - neg_center)
                    weight_value = torch.expm1(weight_value*.7)
                    weight_value = weight_value*patch.abs().max()
                    weights[b, p, ti, 0, :, :] = weight_value.item()
    # 把weights reshape回去(B,num_patches,t, c, h, w)->(B,T,C,H,W)
    token_weights = weights.mean(dim = (-1,-2,-3,-4)) # (b, p)
    weights = pixel_unpatchify(config, weights)
    # 归一化
    weights = weights/weights.sum()

    return weights, token_weights

def count_weight_for_patch_weighted_sparse_mae(config, pixel_values):
    patchified_pixel = pixel_patchify(config, pixel_values)
    B, num_patches, t, c, h, w = patchified_pixel.size()
    # 初始化权重张量
    weights = torch.zeros_like(patchified_pixel)

    # 计算权重
    for b in range(B):
        for p in range(num_patches):
            for ti in range(t):
                patch = patchified_pixel[b, p, ti, 0, :, :]  # (h, w)


                # 找到正值和负值的索引
                pos_indices = torch.nonzero(patch > 0, as_tuple=False).float() # (n1, 2)
                neg_indices = torch.nonzero(patch < 0, as_tuple=False).float() # (n2, 2)
                
                if len(pos_indices) > 1:
                    # 计算正值之间的距离和
                    pos_distances = torch.norm((pos_indices.unsqueeze(0) - pos_indices.unsqueeze(1)), dim=2)
                    pos_distance_sum = pos_distances.mean()
                else:
                    pos_distance_sum = torch.tensor(1e8)
                
                if len(neg_indices) > 1:
                    # 计算负值之间的距离和
                    neg_distances = torch.norm(neg_indices.unsqueeze(0) - neg_indices.unsqueeze(1), dim=2)
                    neg_distance_sum = neg_distances.mean()
                else:
                    neg_distance_sum = torch.tensor(1e8)

                # 计算非零点数
                not_zero_count = len(pos_indices) + len(neg_indices)


                # !将正值和负值的距离和的倒数作为权重
                weight_value = not_zero_count**2/(pos_distance_sum + neg_distance_sum)
                # if weight_value > 1e-1:
                #     print("weight_value",weight_value)
                weights[b, p, ti, 0, :, :] = weight_value.item()

    # 把weights reshape回去(B,num_patches,t, c, h, w)->(B,T,C,H,W)
    token_weights = weights.mean(dim = (-1,-2,-3,-4)) # (b, p)
    weights = pixel_unpatchify(config, weights)
    weights = weights/weights.sum()*h*w
    return weights, token_weights

def count_weight_for_patch_weighted_sparse_expmae(config, pixel_values):
    patchified_pixel = pixel_patchify(config, pixel_values)
    B, num_patches, t, c, h, w = patchified_pixel.size()
    # 初始化权重张量
    weights = torch.zeros_like(patchified_pixel)

    # 计算权重
    for b in range(B):
        for p in range(num_patches):
            for ti in range(t):
                patch = patchified_pixel[b, p, ti, 0, :, :]  # (h, w)
                # 找到正值和负值的索引
                pos_indices = torch.nonzero(patch > 0, as_tuple=False).float() # (18,2)
                neg_indices = torch.nonzero(patch < 0, as_tuple=False).float()
                
                if len(pos_indices) > 1:
                    # 计算正值之间的距离和
                    pos_distances = torch.norm(pos_indices.unsqueeze(0) - pos_indices.unsqueeze(1), dim=2)
                    pos_distance_sum = pos_distances.mean().float()
                    # print("pos_distance_sum",pos_distance_sum)
                else:
                    pos_distance_sum = torch.tensor(1e8)
                
                if len(neg_indices) > 1:
                    # 计算负值之间的距离和
                    neg_distances = torch.norm(neg_indices.unsqueeze(0) - neg_indices.unsqueeze(1), dim=2)
                    neg_distance_sum = neg_distances.mean().float()
                    # print("neg_distance_sum",neg_distance_sum)
                else:
                    neg_distance_sum = torch.tensor(1e8)

                # 计算非零点数
                not_zero_count = len(pos_indices) + len(neg_indices)


                # !将正值和负值的距离和的倒数作为权重
                weight_value =patch.abs().mean()*not_zero_count/(pos_distance_sum + neg_distance_sum)/10
                # weight_value = torch.expm1(1/(pos_distance_sum + neg_distance_sum)).item()
                # if weight_value > 1e-1:
                #     print("weight_value",weight_value)
                
                weights[b, p, ti, 0, :, :] = torch.expm1(weight_value-3.1).item()

    # 把weights reshape回去(B,num_patches,t, c, h, w)->(B,T,C,H,W)
    token_weights = weights.mean(dim = (-1,-2,-3,-4)) # (b, p)
    weights = pixel_unpatchify(config, weights)
    # 把50以下的权重设置为0，并且归一化
    weights[weights<10] = 0
    return weights, token_weights

def count_weight_for_patch_weighted_3sigmamae(config, pixel_values):
    # 计算pixel_values的mu和sigma
    mu = pixel_values.mean()
    sigma = pixel_values.std()
    # print("mu.size",mu.size())
    # print("sigma.size",sigma.size())

    patchified_pixel = pixel_patchify(config, pixel_values) # (B, p, t, c, h, w)
    B, num_patches, t, c, h, w = patchified_pixel.size()
    # 初始化权重张量
    weights = torch.zeros_like(patchified_pixel)

    # 计算权重
    # 统计(B, p, t, c, h, w)后3维度大于3sigma的点的个数
    greater_than_3sigma = (patchified_pixel > mu+3*sigma) | (patchified_pixel < mu-3*sigma)
    # print("greater_than_3sigma size",greater_than_3sigma.size())
    greater_than_3sigma = greater_than_3sigma.float().sum(dim=(3,4,5))
    # print("greater_than_3sigma size after",greater_than_3sigma.size())
    greater_than_3sigma = rearrange(greater_than_3sigma, 'b c h -> b c h 1 1 1')
    weights = greater_than_3sigma.repeat(1, 1, 1, c, h, w)
    # print("weights size (after)",weights.size())

    # 把weights reshape回去(B,num_patches,t, c, h, w)->(B,T,C,H,W)
    token_weights = weights.mean(dim = (-1,-2,-3,-4)) # (b, p)
    weights = pixel_unpatchify(config, weights)
    # 归一化
    weights = weights
    return weights, token_weights

def count_weight_for_patch_weighted_3sigmamae_oncuda(config, pixel_values):
    # 计算pixel_values的mu和sigma
    mu = pixel_values.mean()
    sigma = pixel_values.std()

    patchified_pixel = pixel_patchify(config, pixel_values) # (B, p, t, c, h, w)
    B, num_patches, t, c, h, w = patchified_pixel.size()
    weights = torch.zeros_like(patchified_pixel) # (B, p, t, c, h, w)

    # 计算权重
    # 统计(B, p, t, c, h, w)后3大于3sigma的点的个数 占非零点的比例
    non_zero_count = (patchified_pixel != 0).float().sum(dim=(3,4,5))

    greater_than_3sigma = (patchified_pixel > mu+3*sigma) | (patchified_pixel < mu-3*sigma)
    greater_than_3sigma = greater_than_3sigma.float().sum(dim=(3,4,5))/non_zero_count
    greater_than_3sigma = rearrange(greater_than_3sigma, 'b c h -> b c h 1 1 1')
    weights = greater_than_3sigma.repeat(1, 1, 1, c, h, w)

    # (B,num_patches,t, c, h, w)->(B,T,C,H,W)
    token_weights = weights.mean(dim = (-1,-2,-3,-4)) # (b, p)
    weights = pixel_unpatchify(config, weights)

    # 归一化
    weights = weights/weights.sum()*h*w
    return weights, token_weights

def count_weight_for_patch_weighted_3sigmamae_exp(config, pixel_values):
    # 计算pixel_values的mu和sigma
    mu = pixel_values.mean()
    sigma = pixel_values.std()

    patchified_pixel = pixel_patchify(config, pixel_values)
    B, num_patches, t, c, h, w = patchified_pixel.size()
    # 初始化权重张量
    weights = torch.zeros_like(patchified_pixel)

    # 计算权重
    for b in range(B):
        for p in range(num_patches):
            for ti in range(t):
                patch = patchified_pixel[b, p, ti, 0, :, :]
                # 统计patch中3sigma之外点的个数
                tri_sigma_indices = torch.nonzero((patch > mu + 3*sigma) | (patch < mu - 3*sigma), as_tuple=False).float()
                weight_value =len(tri_sigma_indices) if len(tri_sigma_indices) > 0 else 0.
                
                weigh_value = tri_sigma_indices/h/w

                weights[b, p, ti, 0, :, :] = torch.expm1(weight_value*100)

    # 把weights reshape回去(B,num_patches,t, c, h, w)->(B,T,C,H,W)
    token_weights = weights.mean(dim = (-1,-2,-3,-4)) # (b, p)
    weights = pixel_unpatchify(config, weights)
    # 归一化
    weights = weights
    return weights, token_weights

def map_continous_to_discrete(value, bin = [.01, .05, .12], discrete = [0., 1., 3., 8.]):
    # b np t -> b np t
    weights = torch.zeros_like(value)
    weights[value < bin[0]] = discrete[0]
    weights[(value >= bin[0]) & (value < bin[1])] = discrete[1]
    weights[(value >= bin[1]) & (value < bin[2])] = discrete[2]
    weights[value >= bin[2]] = discrete[3]

    return weights


def count_sgm(pixel_values, sgm_num, tubelet_size = [1, 64, 64], discrete=False):
    # mu, sigma
    mu = pixel_values.mean()
    sigma = pixel_values.std()

    image_size = pixel_values.size()[-1]

    patchified_pixel = pixel_patchify(pixel_values, tubelet_size) # (B, p, c, h, w)
    B, num_patches, c, h, w = patchified_pixel.size()
    weights = torch.zeros_like(patchified_pixel)

    non_zero_count = (patchified_pixel != 0).float().sum(dim=(2,3,4))+1 # (B, p)

    greater_than_3sigma = (patchified_pixel > mu+sgm_num*sigma) | (patchified_pixel < mu-sgm_num*sigma) # (B, p, c, h, w)
    greater_than_3sigma = greater_than_3sigma.float().sum(dim=(2,3,4))/non_zero_count # (B, p)
    if discrete:
        greater_than_3sigma = map_continous_to_discrete(greater_than_3sigma)
    token_weights = greater_than_3sigma
    greater_than_3sigma = rearrange(greater_than_3sigma, 'b np -> b np 1 1 1')
    weights = greater_than_3sigma.repeat(1, 1, c, h, w)

    # (B,num_patches, c, h, w)->(B,C,H,W)
    # token_weights = greater_than_3sigma.mean(dim = (-1,-2,-3,-4)) # (b, p)
    weights = pixel_unpatchify(weights, tubelet_size, image_size)

    weights = (weights/weights.sum())*(image_size**2)
    return weights, token_weights

def count_cv(pixel_values, tubelet_size = [1, 64, 64], discrete=False, weight_order = 'log1p-log1p'):
    image_size = pixel_values.size()[-1]

    if weight_order in ['log1p-log1p', 'log1p-ori']:
        bin = [2., 3., 5.]
        bin_value = [0., 1., 3., 8.]
    elif weight_order in ['ori-log1p', 'ori-ori']:
        bin = [2000., 3000., 6000.]
        bin_value = [0., 1., 3., 8.]

    patchified_pixel = pixel_patchify(pixel_values, tubelet_size) # (B, p, c, h, w)
    B, num_patches, c, h, w = patchified_pixel.size()
    weights = torch.zeros_like(patchified_pixel)

    weights = patchified_pixel.std(dim=(2,3,4)) # b p c h w -> b p 

    if discrete:
        weights = map_continous_to_discrete(weights, bin, bin_value)

    token_weights = weights # (b, p)
    weights = rearrange(weights, 'b np -> b np 1 1 1')
    weights = weights.repeat(1, 1, 1, c, h, w)
    weights = pixel_unpatchify(weights, tubelet_size, image_size)

    weights = (weights/weights.sum())*(image_size**2)*B
    return weights, token_weights

def count_cv_rdbu(pixel_values, tubelet_size = [1, 64, 64], discrete=False):
    image_size = pixel_values.size()[-1]
    patchified_pixel = pixel_patchify(pixel_values, tubelet_size) # (B, p, t, c, h, w)
    B, num_patches, c, h, w = patchified_pixel.size()
    weights = torch.zeros_like(patchified_pixel)

    # 统计patch 的abs的最大值
    max_abs = rearrange(patchified_pixel, 'b p c h w -> b p (c h w)').abs().max(dim=-1, keepdim=False)[0]
    # 统计正值的标准差
    pos_std = torch.where(patchified_pixel > 0, patchified_pixel, 0).std(dim=(2,3,4))
    # 统计负值的标准差
    neg_std = torch.where(patchified_pixel < 0, patchified_pixel, 0).std(dim=(2,3,4))

    weights = max_abs*(pos_std+neg_std)# -> b p t
    weights = weights/weights.max()

    bin, bin_value = ([.37, .52, .75],[0., 4., 8., 12.])

    if discrete:
        weights = map_continous_to_discrete(weights, bin, bin_value)

    token_weights = weights
    weights = rearrange(weights, 'b np -> b np 1 1 1')
    weights = weights.repeat(1, 1, c, h, w)
    weights = pixel_unpatchify(weights, tubelet_size,image_size)

    weights = (weights/weights.sum())*(image_size**2)*B
    return weights, token_weights

def get_weights(loss_patch, pixel_values):
    if loss_patch == "patch_weighted_mae":
        weights, token_weights = count_weight_for_patch_weighted_mae(pixel_values)
    elif loss_patch == "patch_weighted_expmae":
        weights, token_weights = count_weight_for_patch_weighted_expmae(pixel_values)
    elif loss_patch == "patch_weighted_sparse_mae":
        weights, token_weights = count_weight_for_patch_weighted_sparse_mae(pixel_values)
    elif loss_patch == "patch_weighted_sparse_expmae":
        weights, token_weights = count_weight_for_patch_weighted_sparse_expmae(pixel_values)
    elif loss_patch == "patch_weighted_3sigma":
        weights, token_weights = count_weight_for_patch_weighted_3sigmamae(pixel_values)
    elif loss_patch == "patch_weighted_3sigma_exp":
        weights, token_weights = count_weight_for_patch_weighted_3sigmamae_exp(pixel_values)
    elif loss_patch == "3sgm-cuda":
        weights, token_weights = count_weight_for_patch_weighted_3sigmamae_oncuda(pixel_values)
    elif loss_patch == "3sgm-before":
        weights, token_weights = count_weight_for_patch_weighted_3sigmamae(pixel_values)
    elif loss_patch == "4sgm-continous":
        weights, token_weights = count_sgm(pixel_values, 4)
    elif loss_patch == "3.5sgm-continous":
        weights, token_weights = count_sgm(pixel_values, 3.5)
    elif loss_patch == "3sgm-continous":
        weights, token_weights = count_sgm(pixel_values, 3)
    elif loss_patch == "2sgm-continous":
        weights, token_weights = count_sgm(pixel_values, 2)
    elif loss_patch == "1sgm-continous":
        weights, token_weights = count_sgm(pixel_values, 1)
    elif loss_patch == "cv-continous":
        weights, token_weights = count_cv(pixel_values)
    elif loss_patch == "4sgm-discrete":
        weights, token_weights = count_sgm(pixel_values, 4, discrete=True)
    elif loss_patch == "3.5sgm-discrete":
        weights, token_weights = count_sgm(pixel_values, 3.5, discrete=True)
    elif loss_patch == "3sgm-discrete":
        weights, token_weights = count_sgm(pixel_values, 3, discrete=True)
    elif loss_patch == "2sgm-discrete":
        weights, token_weights = count_sgm(pixel_values, 2, discrete=True)
    elif loss_patch == "1sgm-discrete":
        weights, token_weights = count_sgm(pixel_values, 1, discrete=True)
    elif loss_patch == "cv-discrete":
        weights, token_weights = count_cv(pixel_values, discrete=True)
    elif loss_patch == "cv-rdbu":
        weights, token_weights = count_cv_rdbu(pixel_values,discrete=True)
    else:
        raise ValueError(f"loss_patch ^^^{loss_patch}^^^ no released")
    
    if token_weights is None:
        return weights, token_weights
    else:
        # add the cls token weights
        token_weights = torch.cat([torch.zeros_like(token_weights[:,0:1]), token_weights], dim=1) # (b, p) -> (b, p+1)
        return weights, token_weights
    