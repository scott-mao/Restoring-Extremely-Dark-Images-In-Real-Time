import common
import torch.nn as nn

class Net(nn.Module):
    def __init__(self, args, conv=common.default_conv):
        super(Net, self).__init__()

        n_resblocks = args.n_resblocks
        n_feats = args.n_feats
        kernel_size = 3
        scale = args.scale
        act = nn.LeakyReLU(0.2, inplace=False)
        # act = nn.ReLU(True)
        # define head module
        m_head = [conv(args.n_colors, n_feats, kernel_size)]

        # define body module
        m_body = [
            common.ResBlock(
                conv, n_feats, kernel_size, act=act, res_scale=args.res_scale
            ) for _ in range(n_resblocks)
        ]
        m_body.append(conv(n_feats, n_feats, kernel_size))

        # define tail module
        m_tail = [
            common.Upsampler(conv, scale, n_feats, act=False),
            conv(n_feats, args.o_colors, kernel_size)
        ]

        self.head = nn.Sequential(*m_head)
        self.body = nn.Sequential(*m_body)
        self.tail = nn.Sequential(*m_tail)
        
    def downshuffle(self,var,r):
        b,c,h,w = var.size()
        out_channel = c*(r**2)
        out_h = h//r
        out_w = w//r
        return var.contiguous().view(b, c, out_h, r, out_w, r).permute(0,1,3,5,2,4).contiguous().view(b,out_channel, out_h, out_w).contiguous()

    def forward(self, x):
        # x = self.sub_mean(x)
        x = self.head(self.downshuffle(x,2))

        res = self.body(x)
        res += x

        x = self.tail(res)
        # x = self.add_mean(x)

        return x
