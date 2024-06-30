def data_augmentation(img, flip_v, flip_h):
    axis = []
    if flip_v:
        axis.append(2)
    if flip_h:
        axis.append(3)
    if len(axis):
        img = torch.flip(img, axis)
    return img

def flip_denoise(x, denoise_fn, noise_levels, flips=[(False, False)]):
    b, c, w, h = x.shape
    #flips = [(False, False), (True, False), (False, True), (True, True)]
    supports = []
    
    for f in flips:
        supports.append(data_augmentation(x, f[0], f[1]))
    
    x_recon = denoise_fn(torch.cat(supports, dim=0), noise_levels)
    
    split_x_recon = torch.split(x_recon, b, 0)
    
    supports = []
    for idx, f in enumerate(flips):
        supports.append(data_augmentation(split_x_recon[idx], f[0], f[1]).unsqueeze(1))

    x_recon = torch.mean(torch.cat(supports, dim=1), dim=1, keepdim=False)
    return x_recon

def flip_denoise_noise(x, denoise_fn, noise_levels, flips=[(False, False)]):
    b, c, w, h = x.shape
    #flips = [(False, False), (True, False), (False, True), (True, True)]
    supports = []
    
    for f in flips:
        supports.append(data_augmentation(x, f[0], f[1]))
    
    x_recon = denoise_fn(torch.cat(supports, dim=0), noise_levels)
    
    split_x_recon = torch.split(x_recon, b, 0)
    
    supports = []
    for idx, f in enumerate(flips):
        supports.append(data_augmentation(split_x_recon[idx], f[0], f[1]).unsqueeze(1))

    x_recon = torch.mean(torch.cat(supports, dim=1), dim=1, keepdim=False)
    return x_recon



class UNet(nn.Module):
    def __init__(
        self,
        in_channel=6,
        out_channel=3,
        inner_channel=32,
        norm_groups=32,
        channel_mults=(1, 2, 4, 8, 8),
        attn_res=(8),
        res_blocks=3,
        dropout=0,
        with_noise_level_emb=True,
        image_size=128,
        version='v2'
    ):
        super().__init__()
        self.version = version

        if with_noise_level_emb:
            noise_level_channel = inner_channel
            self.noise_level_mlp = nn.Sequential(
                PositionalEncoding(inner_channel),
                nn.Linear(inner_channel, inner_channel * 4),
                Swish(),
                nn.Linear(inner_channel * 4, inner_channel)
            )
        else:
            noise_level_channel = None
            self.noise_level_mlp = None

        num_mults = len(channel_mults)
        pre_channel = inner_channel
        feat_channels = [pre_channel]
        now_res = image_size
        downs = [nn.Conv2d(in_channel, inner_channel,
                           kernel_size=3, padding=1)]

        encoder_dropout = 0.0
        print('dropout', dropout, 'encoder dropout', encoder_dropout)
        for ind in range(num_mults):
            is_last = (ind == num_mults - 1)
            use_attn = (now_res in attn_res)
            channel_mult = inner_channel * channel_mults[ind]
            for _ in range(0, res_blocks):
                downs.append(ResnetBlocWithAttn(
                    pre_channel, channel_mult, noise_level_emb_dim=noise_level_channel, norm_groups=norm_groups, dropout=encoder_dropout, with_attn=use_attn))
                feat_channels.append(channel_mult)
                pre_channel = channel_mult
            if not is_last:
                downs.append(Downsample(pre_channel))
                feat_channels.append(pre_channel)
                now_res = now_res//2
        self.downs = nn.ModuleList(downs)

        self.mid = nn.ModuleList([
            ResnetBlocWithAttn(pre_channel, pre_channel, noise_level_emb_dim=noise_level_channel, norm_groups=norm_groups,
                               dropout=encoder_dropout, with_attn=True),
            ResnetBlocWithAttn(pre_channel, pre_channel, noise_level_emb_dim=noise_level_channel, norm_groups=norm_groups,
                               dropout=encoder_dropout, with_attn=False)
        ])

        ups = []
        for ind in reversed(range(num_mults)):
            is_last = (ind < 1)
            use_attn = (now_res in attn_res)
            channel_mult = inner_channel * channel_mults[ind]
            for _ in range(0, res_blocks+1):
                ups.append(ResnetBlocWithAttn(
                    pre_channel+feat_channels.pop(), channel_mult, noise_level_emb_dim=noise_level_channel, norm_groups=norm_groups,
                        dropout=dropout, with_attn=use_attn))
                pre_channel = channel_mult
            if not is_last:
                ups.append(Upsample(pre_channel))
                now_res = now_res*2

        self.ups = nn.ModuleList(ups)

        if version == 'v2':
            self.final_conv1 = Block(pre_channel, pre_channel, groups=norm_groups, additional_dim=in_channel)
            self.final_conv2 = Block(pre_channel, default(out_channel, in_channel), groups=norm_groups)
        else:
            self.final_conv1 = Block(pre_channel, default(out_channel, in_channel), groups=norm_groups)


    def forward(self, x, time=None):
        t = self.noise_level_mlp(time) if exists(
            self.noise_level_mlp) else None

        feats = [x]
        for layer in self.downs:
            if isinstance(layer, ResnetBlocWithAttn):
                x = layer(x, t)
            else:
                x = layer(x)
            feats.append(x)

        for layer in self.mid:
            if isinstance(layer, ResnetBlocWithAttn):
                x = layer(x, t)
            else:
                x = layer(x)

        for layer in self.ups:
            if isinstance(layer, ResnetBlocWithAttn):
                skip = feats.pop()
                x = F.interpolate(x, size=skip.shape[-2:])
                x = layer(torch.cat((x, skip), dim=1), t)
            else:
                x = layer(x)
        #x = torch.cat((x, feats.pop()), dim=1)

        if self.version == 'v2':
            noise = self.final_conv1(x, feats.pop())
            noise = self.final_conv2(noise)
        else:
            noise = self.final_conv1(x)
            
        # test!
        # noise = torch.tanh(noise)

        return noise
