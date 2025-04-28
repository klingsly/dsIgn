class DSIgnn(nn.Module):
    def __init__(self, n_inputs1, n_inputs2, n_hiddens, n_outputs, d_model=128, n_head=4, dropout=0.2):
        super(DSIgnn, self).__init__()

        self.dropout = dropout
        self.hgc1 = HGNNconv(n_inputs1, n_hiddens)
        self.hgc2 = HGNNconv(n_hiddens, n_rois)

        self.input_embedding = nn.Linear(n_inputs2, d_model)
        self.encoder = nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model, n_head, dropout=dropout), 2)
        self.decoder = KAN([d_model, n_snps])

        self.fc1 = nn.Linear(n_rois+n_snps, n_outputs)
        self.fc2 = nn.Linear(n_outputs, 1)

        self.input_embedding.apply(initialization)
        self.fc1.apply(initialization)
        self.fc2.apply(initialization)

        self.loss = CCA(n_rois).loss

    def forward(self, x, g, y):
        x = F.leaky_relu(self.hgc1(x, g))
        x = F.dropout(x, self.dropout)
        x = self.hgc2(x, g)

        y = self.input_embedding(y)
        y = y.unsqueeze(1)
        y = self.encoder(y)
        y = y.squeeze(1)
        y = self.decoder(y)

        output = torch.cat([x, y], dim=1)
        output = F.relu(self.fc1(output))
        output = self.fc2(output)

        cor_loss = self.loss(x, y)
        return output, cor_loss


class HGNNconv(nn.Module):
    def __init__(self, in_ft, out_ft, bias=True):
        super(HGNNconv, self).__init__()

        self.weight = Parameter(torch.Tensor(in_ft, out_ft))
        if bias:
            self.bias = Parameter(torch.Tensor(out_ft))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, x: torch.Tensor, G: torch.Tensor):
        x = x.matmul(self.weight)
        if self.bias is not None:
            x = x + self.bias
        x = G.matmul(x)
        return x


class CCA():
    def __init__(self, n_outputs, use_all_singular_values=False):
        self.outdim_size = n_outputs
        self.use_all_singular_values = use_all_singular_values
        self.device = device

    def loss(self, H1, H2):
        r1 = 1e-3
        r2 = 1e-3
        eps = 1e-8

        H1, H2 = H1.t(), H2.t()

        o = H1.size(0)
        m = H1.size(1)

        # ȥ���Ļ������ֵ��
        H1bar = H1 - H1.mean(dim=1).unsqueeze(dim=1)
        H2bar = H2 - H2.mean(dim=1).unsqueeze(dim=1)

        # ����Sxy,Sxx = 1/(m-1)*H1+r1*I, Syy = 1/(m-1)*H2+r2*I
        Sxy = (1.0 / (m - 1)) * torch.matmul(H1bar, H2bar.t())
        Sxx = (1.0 / (m - 1)) * torch.matmul(H1bar, H1bar.t()) + r1 * torch.eye(o, device=self.device)
        Syy = (1.0 / (m - 1)) * torch.matmul(H2bar, H2bar.t()) + r2 * torch.eye(o, device=self.device)

        # ʹ�������ֽ����Э�������ĸ���
        [D1, V1] = torch.linalg.eigh(Sxx)
        [D2, V2] = torch.linalg.eigh(Syy)

        # ����ȶ��ԣ�����������ֵ
        idx = torch.gt(D1, eps).nonzero()[:, 0]
        D1 = D1[idx]
        V1 = V1[:, idx]
        idx = torch.gt(D2, eps).nonzero()[:, 0]
        D2 = D2[idx]
        V2 = V2[:, idx]

        # ����Sxx,Syy��-1/2�η�
        SxxInv = torch.matmul((torch.matmul(V1, torch.diag(D1 ** -0.5))), V1.t())
        SyyInv = torch.matmul((torch.matmul(V2, torch.diag(D2 ** -0.5))), V2.t())

        M = torch.matmul(torch.matmul(SxxInv, Sxy), SyyInv)

        if self.use_all_singular_values:
            tmp = torch.matmul(M.t(), M)
            corr = torch.trace(torch.sqrt(tmp))
        else:
            trace = torch.matmul(M.t(), M)
            trace = torch.add(trace, (torch.eye(trace.shape[0]) * r1).to(self.device))
            U, V = torch.linalg.eigh(trace)
            U = torch.where(U > eps, U, (torch.ones(U.shape).float() * eps).to(self.device))
            U = U.topk(self.outdim_size)[0]
            corr = torch.sum(torch.sqrt(U))
        return -corr


class KAN(torch.nn.Module):
    def __init__(self, layers_hidden, grid_size=5, spline_order=3, scale_noise=0.1, scale_base=1.0, scale_spline=1.0,
                 base_activation=torch.nn.SiLU, grid_eps=0.02, grid_range=[-1, 1],):
        super(KAN, self).__init__()
        self.grid_size = grid_size
        self.spline_order = spline_order

        self.layers = torch.nn.ModuleList()
        for in_features, out_features in zip(layers_hidden, layers_hidden[1:]):
            self.layers.append(
                KANLinear(in_features, out_features, grid_size=grid_size, spline_order=spline_order,
                          scale_noise=scale_noise, scale_base=scale_base, scale_spline=scale_spline,
                          base_activation=base_activation, grid_eps=grid_eps, grid_range=grid_range)
            )

    def forward(self, x: torch.Tensor, update_grid=False):
        for layer in self.layers:
            if update_grid:
                layer.update_grid(x)
            x = layer(x)
        return x

    def regularization_loss(self, regularize_activation=1.0, regularize_entropy=1.0):
        # ����������ʧ�ķ���������Լ��ģ�͵Ĳ�������ֹ����ϡ�
        return sum(
            layer.regularization_loss(regularize_activation, regularize_entropy)
            for layer in self.layers
        )


class KANLinear(torch.nn.Module):
    def __init__(
        self,
        in_features,
        out_features,
        grid_size=5,  # �����С��Ĭ��Ϊ 5
        spline_order=3, # �ֶζ���ʽ�Ľ�����Ĭ��Ϊ 3
        scale_noise=0.1,  # ����������Ĭ��Ϊ 0.1
        scale_base=1.0,   # �������ţ�Ĭ��Ϊ 1.0
        scale_spline=1.0,    # �ֶζ���ʽ�����ţ�Ĭ��Ϊ 1.0
        enable_standalone_scale_spline=True,
        base_activation=torch.nn.SiLU,  # �����������Ĭ��Ϊ SiLU��Sigmoid Linear Unit��
        grid_eps=0.02,
        grid_range=[-1, 1],  # ����Χ��Ĭ��Ϊ [-1, 1]
    ):
        super(KANLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.grid_size = grid_size  # ���������С�ͷֶζ���ʽ�Ľ���
        self.spline_order = spline_order

        h = (grid_range[1] - grid_range[0]) / grid_size   # �������񲽳�
        grid = ( # ��������
            (
                torch.arange(-spline_order, grid_size + spline_order + 1) * h
                + grid_range[0]
            )
            .expand(in_features, -1)
            .contiguous()
        )
        self.register_buffer("grid", grid)  # ��������Ϊ������ע��

        self.base_weight = torch.nn.Parameter(torch.Tensor(out_features, in_features)) # ��ʼ������Ȩ�غͷֶζ���ʽȨ��
        self.spline_weight = torch.nn.Parameter(
            torch.Tensor(out_features, in_features, grid_size + spline_order)
        )
        if enable_standalone_scale_spline:  # ������ö����ķֶζ���ʽ���ţ����ʼ���ֶζ���ʽ���Ų���
            self.spline_scaler = torch.nn.Parameter(
                torch.Tensor(out_features, in_features)
            )

        self.scale_noise = scale_noise  # ���������������������š��ֶζ���ʽ�����š��Ƿ����ö����ķֶζ���ʽ���š����������������Χ���ݲ�
        self.scale_base = scale_base
        self.scale_spline = scale_spline
        self.enable_standalone_scale_spline = enable_standalone_scale_spline
        self.base_activation = base_activation()
        self.grid_eps = grid_eps

        self.reset_parameters()  # ���ò���

    def reset_parameters(self):
        torch.nn.init.kaiming_uniform_(self.base_weight, a=math.sqrt(5) * self.scale_base)# ʹ�� Kaiming ���ȳ�ʼ������Ȩ��
        with torch.no_grad():
            noise = (# ������������
                (
                    torch.rand(self.grid_size + 1, self.in_features, self.out_features)
                    - 1 / 2
                )
                * self.scale_noise
                / self.grid_size
            )
            self.spline_weight.data.copy_( # ����ֶζ���ʽȨ��
                (self.scale_spline if not self.enable_standalone_scale_spline else 1.0)
                * self.curve2coeff(
                    self.grid.T[self.spline_order : -self.spline_order],
                    noise,
                )
            )
            if self.enable_standalone_scale_spline:  # ������ö����ķֶζ���ʽ���ţ���ʹ�� Kaiming ���ȳ�ʼ���ֶζ���ʽ���Ų���
                # torch.nn.init.constant_(self.spline_scaler, self.scale_spline)
                torch.nn.init.kaiming_uniform_(self.spline_scaler, a=math.sqrt(5) * self.scale_spline)

    def b_splines(self, x: torch.Tensor):
        assert x.dim() == 2 and x.size(1) == self.in_features

        grid: torch.Tensor = ( # ��״Ϊ (in_features, grid_size + 2 * spline_order + 1)
            self.grid
        )  # (in_features, grid_size + 2 * spline_order + 1)
        x = x.unsqueeze(-1)
        bases = ((x >= grid[:, :-1]) & (x < grid[:, 1:])).to(x.dtype)
        for k in range(1, self.spline_order + 1):
            bases = (
                (x - grid[:, : -(k + 1)])
                / (grid[:, k:-1] - grid[:, : -(k + 1)])
                * bases[:, :, :-1]
            ) + (
                (grid[:, k + 1 :] - x)
                / (grid[:, k + 1 :] - grid[:, 1:(-k)])
                * bases[:, :, 1:]
            )

        assert bases.size() == (
            x.size(0),
            self.in_features,
            self.grid_size + self.spline_order,
        )
        return bases.contiguous()

    def curve2coeff(self, x: torch.Tensor, y: torch.Tensor):
        assert x.dim() == 2 and x.size(1) == self.in_features
        assert y.size() == (x.size(0), self.in_features, self.out_features)
        # ���� B-����������
        A = self.b_splines(x).transpose(
            0, 1 # ��״Ϊ (in_features, batch_size, grid_size + spline_order)
        )  # (in_features, batch_size, grid_size + spline_order)
        B = y.transpose(0, 1)  # (in_features, batch_size, out_features) # ��״Ϊ (in_features, batch_size, out_features)
        solution = torch.linalg.lstsq(   # ʹ����С���˷�������Է�����
            A, B
        ).solution  # (in_features, grid_size + spline_order, out_features)  # ��״Ϊ (in_features, grid_size + spline_order, out_features)
        result = solution.permute( # ���������ά��˳��
            2, 0, 1
        )  # (out_features, in_features, grid_size + spline_order)

        assert result.size() == (
            self.out_features,
            self.in_features,
            self.grid_size + self.spline_order,
        )
        return result.contiguous()

    @property
    def scaled_spline_weight(self):
        return self.spline_weight * (
            self.spline_scaler.unsqueeze(-1)
            if self.enable_standalone_scale_spline
            else 1.0
        )

    def forward(self, x: torch.Tensor):
        # ����������ͨ��ģ�͵ĸ����㣬�������Ա任�ͼ�����������յõ�ģ�͵�������
        assert x.dim() == 2 and x.size(1) == self.in_features

        base_output = F.linear(self.base_activation(x), self.base_weight) # ����������Բ�����
        spline_output = F.linear( # ����ֶζ���ʽ���Բ�����
            self.b_splines(x).view(x.size(0), -1),
            self.scaled_spline_weight.view(self.out_features, -1),
        )
        return base_output + spline_output  # ���ػ������Բ�����ͷֶζ���ʽ���Բ�����ĺ�

    @torch.no_grad()
    # ��������
    # ����:
    # x (torch.Tensor): ������������״Ϊ (batch_size, in_features)��
    # margin (float): �����Ե�հ׵Ĵ�С��Ĭ��Ϊ 0.01��
    # ������������ x �ķֲ��������̬����ģ�͵�����,ʹ��ģ���ܹ����õ���Ӧ�������ݵķֲ��ص㣬�Ӷ����ģ�͵ı�������ͷ���������
    def update_grid(self, x: torch.Tensor, margin=0.01):
        assert x.dim() == 2 and x.size(1) == self.in_features
        batch = x.size(0)

        splines = self.b_splines(x)  # (batch, in, coeff)  # ���� B-����������
        splines = splines.permute(1, 0, 2)  # (in, batch, coeff)  # ����ά��˳��Ϊ (in, batch, coeff)
        orig_coeff = self.scaled_spline_weight  # (out, in, coeff)
        orig_coeff = orig_coeff.permute(1, 2, 0)  # (in, coeff, out)  # ����ά��˳��Ϊ (in, coeff, out)
        unreduced_spline_output = torch.bmm(splines, orig_coeff)  # (in, batch, out)
        unreduced_spline_output = unreduced_spline_output.permute(
            1, 0, 2
        )  # (batch, in, out)

        # sort each channel individually to collect data distribution
        x_sorted = torch.sort(x, dim=0)[0] # ��ÿ��ͨ�������������ռ����ݷֲ�
        grid_adaptive = x_sorted[
            torch.linspace(
                0, batch - 1, self.grid_size + 1, dtype=torch.int64, device=x.device
            )
        ]

        uniform_step = (x_sorted[-1] - x_sorted[0] + 2 * margin) / self.grid_size
        grid_uniform = (
            torch.arange(
                self.grid_size + 1, dtype=torch.float32, device=x.device
            ).unsqueeze(1)
            * uniform_step
            + x_sorted[0]
            - margin
        )

        grid = self.grid_eps * grid_uniform + (1 - self.grid_eps) * grid_adaptive
        grid = torch.concatenate(
            [
                grid[:1]
                - uniform_step
                * torch.arange(self.spline_order, 0, -1, device=x.device).unsqueeze(1),
                grid,
                grid[-1:]
                + uniform_step
                * torch.arange(1, self.spline_order + 1, device=x.device).unsqueeze(1),
            ],
            dim=0,
        )

        self.grid.copy_(grid.T)   # ��������ͷֶζ���ʽȨ��
        self.spline_weight.data.copy_(self.curve2coeff(x, unreduced_spline_output))

    def regularization_loss(self, regularize_activation=1.0, regularize_entropy=1.0):
        # ����������ʧ������Լ��ģ�͵Ĳ�������ֹ�����
        l1_fake = self.spline_weight.abs().mean(-1)
        regularization_loss_activation = l1_fake.sum()
        p = l1_fake / regularization_loss_activation
        regularization_loss_entropy = -torch.sum(p * p.log())
        return (
            regularize_activation * regularization_loss_activation
            + regularize_entropy * regularization_loss_entropy
        )