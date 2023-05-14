from models.basic_modules import *
from models.basic_modules import inconv



class reconAE(nn.Module):
    def __init__(self, num_in_ch, seq_len, features_root, skip_ops):
        super(reconAE, self).__init__()
        self.num_in_ch = num_in_ch
        self.seq_len = seq_len
        self.skip_ops = skip_ops

        self.in_conv = inconv(num_in_ch * seq_len, features_root)
        self.down_1 = down(features_root, features_root * 2)
        self.down_2 = down(features_root * 2, features_root * 4)
        self.down_3 = down(features_root * 4, features_root * 8)
        self.down_4 = down(features_root * 8, features_root * 16)

        self.up_4 = up(features_root * 16, features_root * 8, op=self.skip_ops[-1])
        self.up_3 = up(features_root * 8, features_root * 4, op=self.skip_ops[-2])
        self.up_2 = up(features_root * 4, features_root * 2, op=self.skip_ops[-3])
        self.up_1 = up(features_root * 2, features_root, op=self.skip_ops[-4])
        self.out_conv = outconv(features_root, num_in_ch * seq_len)
        
        self.upsample=nn.Upsample(scale_factor=4, mode='nearest')

    def forward(self, x):
        """
        :param x: size [bs,C*seq_len,H,W]
        :return:
        """
        #x=self.upsample(x)
        
        x0 = self.in_conv(x)
        x1 = self.down_1(x0)
        x2 = self.down_2(x1)
        x3 = self.down_3(x2)
        x4 = self.down_4(x3)

        recon = self.up_4(x4, x3 if self.skip_ops[-1] != "none" else None)
        recon = self.up_3(recon, x2 if self.skip_ops[-2] != "none" else None)
        recon = self.up_2(recon, x1 if self.skip_ops[-3] != "none" else None)
        recon = self.up_1(recon, x0 if self.skip_ops[-4] != "none" else None)
        recon = self.out_conv(recon)

        outs=recon
        
        return x,outs


if __name__ == '__main__':
    model = reconAE(num_in_ch=2, seq_len=1, features_root=16,skip_ops=["none", "concat", "concat", "concat"])
    print(model)
    dummy_x = torch.rand(4, 2, 64, 64)
    x,dummy_out = model(dummy_x)
    print(x.shape)
    print(dummy_out.shape)
    print(-1)