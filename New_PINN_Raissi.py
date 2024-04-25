import torch, os

from datetime import datetime

from torch.utils.tensorboard import SummaryWriter
from torch.autograd import grad

NU = 0.01

# Check for CUDA availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class New_PINN_Raissi(torch.nn.Module):
    def __init__(self, df_train, mask_inlet, mask_sdf, mask_top, mask_bot):
        super(New_PINN_Raissi, self).__init__()
        self.x = torch.tensor(df_train['x'].astype(float).values, requires_grad=True).float().unsqueeze(1).to(device)
        self.y = torch.tensor(df_train['y'].astype(float).values, requires_grad=True).float().unsqueeze(1).to(device)
        self.sdf = torch.tensor(df_train['sdf'].astype(float).values).float().unsqueeze(1).to(device)

        self.u = torch.tensor(df_train['u'].astype(float).values).float().unsqueeze(1).to(device)
        self.v = torch.tensor(df_train['v'].astype(float).values).float().unsqueeze(1).to(device)

        self.mask_inlet = mask_inlet
        self.mask_sdf = mask_sdf
        self.mask_top = mask_top
        self.mask_bot = mask_bot

        self.layers = [3, 256, 256, 256, 256, 256, 256, 256, 256, 256, 3]
        print(f"layers: {self.layers}")
        self.model = self.create_model(self.layers)

        self.mse_loss = torch.nn.MSELoss()

        self.lbfgs_optimizer = torch.optim.LBFGS([{'params': self.model.parameters()}], line_search_fn='strong_wolfe') 
        self.writer = SummaryWriter(log_dir=f"logs/{datetime.now().strftime('%Y%m%d-%H%M%S')}")    

    def create_model(self, type_layers):
        layers = []
        for i in range(len(type_layers) - 1):
            layers.append(torch.nn.Linear(type_layers[i], type_layers[i+1]))
            if i != len(type_layers) - 2:
                layers.append(torch.nn.Softplus(100))
        return torch.nn.Sequential(*layers)

    def net_NS(self):
        inputs = torch.cat([self.x, self.y, self.sdf], dim=1)
        output = self.model(inputs)
        u = output[:, 0:1]
        v = output[:, 1:2]
        p = output[:, 2:3]

        u_x = grad(u, self.x, grad_outputs=torch.ones_like(u), create_graph=True)[0]
        u_y = grad(u, self.y, grad_outputs=torch.ones_like(u), create_graph=True)[0]
        v_x = grad(v, self.x, grad_outputs=torch.ones_like(v), create_graph=True)[0]
        v_y = grad(v, self.y, grad_outputs=torch.ones_like(v), create_graph=True)[0]
        u_xx = grad(u_x, self.x, grad_outputs=torch.ones_like(u_x), create_graph=True)[0]
        u_yy = grad(u_y, self.y, grad_outputs=torch.ones_like(u_y), create_graph=True)[0]
        v_xx = grad(v_x, self.x, grad_outputs=torch.ones_like(v_x), create_graph=True)[0]
        v_yy = grad(v_y, self.y, grad_outputs=torch.ones_like(v_y), create_graph=True)[0]
        p_x = grad(p, self.x, grad_outputs=torch.ones_like(p), create_graph=True)[0]
        p_y = grad(p, self.y, grad_outputs=torch.ones_like(p), create_graph=True)[0]

        f_u = (v * u_y - u * v_y) + p_x - NU * (u_xx + u_yy)
        f_v = (u * v_x - v * u_x) + p_y - NU * (v_xx + v_yy)
        ic = u_x + v_y
        
        return u, v, p, f_u, f_v, ic
    
    def forward(self):
        u_pred, v_pred, _, f_u_pred, f_v_pred, ic_pred = self.net_NS()

        u_bc_loss = self.mse_loss(self.u[self.mask_inlet], u_pred[self.mask_inlet]) + self.mse_loss(self.u[self.mask_sdf], u_pred[self.mask_sdf]) + self.mse_loss(u_pred[self.mask_top], u_pred[self.mask_bot])
        v_bc_loss = self.mse_loss(self.v[self.mask_inlet], v_pred[self.mask_inlet]) + self.mse_loss(self.v[self.mask_sdf], v_pred[self.mask_sdf]) + self.mse_loss(v_pred[self.mask_top], -v_pred[self.mask_bot])

        rans_loss = self.mse_loss(f_u_pred, torch.zeros_like(f_u_pred)) + self.mse_loss(f_v_pred, torch.zeros_like(f_v_pred))
        ic_loss = self.mse_loss(ic_pred, torch.zeros_like(ic_pred))
        loss = u_bc_loss + v_bc_loss + rans_loss + ic_loss
        
        return loss, u_bc_loss, v_bc_loss, rans_loss, ic_loss
    
    def train(self, nIter, checkpoint_path='path_to_checkpoint.pth'):
        self.display = {}
        self.temp_losses = {}
        loss_not_diminished_counter = 0
        last_loss = float('inf')

        start_iteration = self.load_checkpoint(checkpoint_path)

        def compute_losses():
            loss, u_bc_loss, v_bc_loss, rans_loss, ic_loss = self.forward()
            self.display = {
                'u_bc_loss': u_bc_loss, 'v_bc_loss': v_bc_loss,
                'rans_loss': rans_loss, 'ic_loss': ic_loss,
            }
            self.temp_losses = {'loss': loss}

        for it in range(start_iteration, nIter + start_iteration):
            def closure():
                self.lbfgs_optimizer.zero_grad()
                compute_losses()
                self.temp_losses['loss'].backward()
                return self.temp_losses['loss']

            self.lbfgs_optimizer.step(closure)

            current_loss = self.temp_losses['loss'].item()
            if current_loss >= last_loss:
                loss_not_diminished_counter += 1
            else:
                loss_not_diminished_counter = 0
            last_loss = current_loss

            if loss_not_diminished_counter >= 10:
                print(f"Stopping early at iteration {it} due to no improvement.")
                break

            if it % 2 == 0: 
                print(f'It: {it}')
            if it % 10 == 0:
                for name, value in self.display.items():
                    print(f"{name}: {value.item()}")
                self.save_checkpoint(checkpoint_path, it)

    def predict(self, df_test):
        x_star = torch.tensor(df_test['x'].astype(float).values, requires_grad=True).float().unsqueeze(1).to(device)
        y_star = torch.tensor(df_test['y'].astype(float).values, requires_grad=True).float().unsqueeze(1).to(device)
        sdf_star = torch.tensor(df_test['sdf'].astype(float).values, requires_grad=True).float().unsqueeze(1).to(device)

        inputs = torch.cat([x_star, y_star, sdf_star], dim=1)
        output = self.model(inputs)
        u_star = output[:, 0:1]
        v_star = output[:, 1:2]
        p_star = output[:, 2:3]
        
        return u_star.cpu().detach().numpy(), v_star.cpu().detach().numpy(), p_star.cpu().detach().numpy()


    def load_checkpoint(self, checkpoint_path): 
        if os.path.isfile(checkpoint_path):
            print(f"Loading checkpoint '{checkpoint_path}'")
            checkpoint = torch.load(checkpoint_path)
            
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.lbfgs_optimizer.load_state_dict(checkpoint['lbfgs_optimizer_state_dict'])
                        
            # Restore the RNG state
            torch.set_rng_state(checkpoint['rng_state'])

            # If you're resuming training and want to start from the next iteration,
            # make sure to load the last iteration count and add one
            start_iteration = checkpoint.get('iterations', 0) + 1
            print(f"Resuming from iteration {start_iteration}")
        else:
            print(f"No checkpoint found at '{checkpoint_path}', starting from scratch.")
            start_iteration = 0

        return start_iteration

    def save_checkpoint(self, checkpoint_path, it):
        if os.path.exists(checkpoint_path):
            os.remove(checkpoint_path)
        
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'lbfgs_optimizer_state_dict': self.lbfgs_optimizer.state_dict(),

            'iterations': it,

            'rng_state': torch.get_rng_state(),
        }

        torch.save(checkpoint, checkpoint_path)
        print(f"Saved checkpoint to '{checkpoint_path}' at iteration {it}")
