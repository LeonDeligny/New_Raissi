import torch, os

import numpy as np

from datetime import datetime

from torch.utils.tensorboard import SummaryWriter
from torch.autograd import grad

import torch.optim as optim
from torch.optim.lr_scheduler import ExponentialLR
from torch.utils.data import Dataset, DataLoader
import fourier_neural_operator.fourier_2d as fourier_2d 


# Properties of air at sea level and 293.15K
RHO = 1.184
NU = 1.56e-5
C = 346.1
P_ref = 1.013e5


# Check for CUDA availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class FourierNeuralOperatorNN(torch.nn.Module):
    def __init__(self, df_train, bc_mask):
        super(FourierNeuralOperatorNN, self).__init__()
        self.x = torch.tensor(df_train['x'].astype(float).values, requires_grad=True).float().unsqueeze(1).to(device)
        self.y = torch.tensor(df_train['y'].astype(float).values, requires_grad=True).float().unsqueeze(1).to(device)
    
        self.u = torch.tensor(df_train['u'].astype(float).values).float().unsqueeze(1).to(device)
        self.v = torch.tensor(df_train['v'].astype(float).values).float().unsqueeze(1).to(device)
        self.p = torch.tensor(df_train['p'].astype(float).values).float().unsqueeze(1).to(device)

        self.bc_mask = bc_mask

        self.u_net = fourier_2d.FNO2d(channel_input=2, output_channel=2, modes1=12, modes2=12, width=50)
        self.v_net = fourier_2d.FNO2d(channel_input=2, output_channel=2, modes1=12, modes2=12, width=50)
        self.p_net = fourier_2d.FNO2d(channel_input=2, output_channel=2, modes1=12, modes2=12, width=50)

        self.u_optimizer = optim.Adam(self.u_net.parameters(), lr=0.001)
        self.v_optimizer = optim.Adam(self.v_net.parameters(), lr=0.001)
        self.p_optimizer = optim.Adam(self.p_net.parameters(), lr=0.001)

        self.u_scheduler = ExponentialLR(self.u_optimizer, gamma=0.95)
        self.v_scheduler = ExponentialLR(self.v_optimizer, gamma=0.95)
        self.p_scheduler = ExponentialLR(self.p_optimizer, gamma=0.95)

        self.loss_func = torch.nn.MSELoss()

    
    def net_NS(self, x, y):
        inputs = torch.cat([x, y], dim=1)
        x_y_grid = inputs.view(1, 2, 2500, 2)

        u = self.u_net(x_y_grid)
        v = self.v_net(x_y_grid)
        p = self.p_net(x_y_grid)

        u_x = grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True)[0]
        u_y = grad(u, y, grad_outputs=torch.ones_like(u), create_graph=True)[0]
        u_xx = grad(u_x, x, grad_outputs=torch.ones_like(u_x), create_graph=True)[0]
        u_yy = grad(u_y, y, grad_outputs=torch.ones_like(u_y), create_graph=True)[0]

        v_x = grad(v, x, grad_outputs=torch.ones_like(v), create_graph=True)[0]
        v_y = grad(v, y, grad_outputs=torch.ones_like(v), create_graph=True)[0]
        v_xx = grad(v_x, x, grad_outputs=torch.ones_like(v_x), create_graph=True)[0]
        v_yy = grad(v_y, y, grad_outputs=torch.ones_like(v_y), create_graph=True)[0]

        p_x = grad(p, x, grad_outputs=torch.ones_like(p), create_graph=True)[0]
        p_y = grad(p, y, grad_outputs=torch.ones_like(p), create_graph=True)[0]

        f_u = (v * u_y - u * v_y) + p_x - NU * (u_xx + u_yy)
        f_v = (u * v_x - v * u_x) + p_y - NU * (v_xx + v_yy)
        ic = u_x + v_y

        return u, v, p, f_u, f_v, ic

    def forward(self, x, y):
        u_pred, v_pred, p_pred, f_u_pred, f_v_pred, ic_pred = self.net_NS(x, y)

        u_bc_loss = self.loss_func(self.u[self.bc_mask], u_pred[self.bc_mask])
        v_bc_loss = self.loss_func(self.v[self.bc_mask], v_pred[self.bc_mask])
        p_bc_loss = self.loss_func(self.p[self.bc_mask], p_pred[self.bc_mask])

        f_u_loss, f_v_loss = self.loss_func(f_u_pred, torch.zeros_like(f_u_pred)), self.loss_func(f_v_pred, torch.zeros_like(f_v_pred))
        ic_loss = self.loss_func(ic_pred, torch.zeros_like(ic_pred)),

        rans_loss = f_u_loss + f_v_loss

        ic_loss = self.loss_func(ic_pred, torch.zeros_like(ic_pred))

        u_loss = u_bc_loss + rans_loss + ic_loss 
        v_loss = v_bc_loss + rans_loss + ic_loss
        p_loss = p_bc_loss + rans_loss + ic_loss

        return u_loss, v_loss, p_loss, u_bc_loss, v_bc_loss, p_bc_loss, f_u_loss, f_v_loss, ic_loss

    def train(self, nIter, checkpoint_path='path_to_checkpoint.pth'):
        # Temporary storage for loss values for logging purposes
        self.temp_losses = {}
        self.display = {}

        if os.path.isfile(checkpoint_path):
            print(f"Loading checkpoint '{checkpoint_path}'")
            checkpoint = torch.load(checkpoint_path)
            
            self.u_net.load_state_dict(checkpoint['u_net_state_dict'])
            self.u_optimizer.load_state_dict(checkpoint['u_optimizer_state_dict'])
            self.u_scheduler.load_state_dict(checkpoint['u_scheduler_state_dict'])

            self.v_net.load_state_dict(checkpoint['v_net_state_dict'])
            self.v_optimizer.load_state_dict(checkpoint['v_optimizer_state_dict'])
            self.v_scheduler.load_state_dict(checkpoint['v_scheduler_state_dict'])

            self.p_net.load_state_dict(checkpoint['p_net_state_dict'])
            self.p_optimizer.load_state_dict(checkpoint['p_optimizer_state_dict'])
            self.p_scheduler.load_state_dict(checkpoint['p_scheduler_state_dict'])

            # Restore the RNG state
            torch.set_rng_state(checkpoint['rng_state'])

            # If you're resuming training and want to start from the next iteration,
            # make sure to load the last iteration count and add one
            start_iteration = checkpoint.get('iterations', 0) + 1
            print(f"Resuming from iteration {start_iteration}")
        else:
            print(f"No checkpoint found at '{checkpoint_path}', starting from scratch.")
            start_iteration = 0

        def compute_losses():
            # Compute all losses
            losses = self.forward(self.x, self.y)

            # Unpack the losses and store them in a dictionary for easy access
            (u_loss, v_loss, p_loss, u_bc_loss, v_bc_loss, p_bc_loss, f_u_loss, f_v_loss, ic_loss) = losses

            self.temp_losses = {'u_loss': u_loss, 'v_loss': v_loss, 'p_loss': p_loss}

            self.display = {
                            'u_loss': u_loss, 'v_loss': v_loss, 'p_loss': p_loss,
                            'u_bc_loss': u_bc_loss, 'v_bc_loss': v_bc_loss, 'p_bc_loss': p_bc_loss,
                            'f_u_loss': f_u_loss, 'f_v_loss': f_v_loss, 'ic_loss': ic_loss
                        }


        for it in range(start_iteration, nIter + start_iteration):
                
            compute_losses()

            self.u_optimizer.zero_grad()
            self.temp_losses['u_loss'].backward()
            self.u_optimizer.step()
            self.u_scheduler.step()

            self.v_optimizer.zero_grad()
            self.temp_losses['v_loss'].backward()
            self.v_optimizer.step()
            self.v_scheduler.step()

            self.p_optimizer.zero_grad()
            self.temp_losses['p_loss'].backward()
            self.p_optimizer.step()
            self.p_scheduler.step()

            if it % 2 == 0:
                print(f"Iteration: {it}")
            if it % 10 == 0:  # Print losses every 10 iterations
                for name, value in self.display.items():
                    print(f"{name}: {value.item()}")

                if os.path.exists(checkpoint_path):
                    os.remove(checkpoint_path)
                
                checkpoint = {
                    'u_net_state_dict': self.u_net.state_dict(),
                    'u_optimizer_state_dict': self.u_optimizer.state_dict(),
                    'u_scheduler_state_dict': self.u_scheduler.state_dict(),

                    'v_net_state_dict': self.v_net.state_dict(),
                    'v_optimizer_state_dict': self.v_optimizer.state_dict(),
                    'v_scheduler_state_dict': self.v_scheduler.state_dict(),

                    'p_net_state_dict': self.p_net.state_dict(),
                    'p_optimizer_state_dict': self.p_optimizer.state_dict(),
                    'p_scheduler_state_dict': self.p_scheduler.state_dict(),

                    'iterations': it,

                    'rng_state': torch.get_rng_state(),
                }

                torch.save(checkpoint, checkpoint_path)
                print(f"Saved checkpoint to '{checkpoint_path}' at iteration {it}")


    def predict(self, df_test):
        x_star = torch.tensor(df_test['x'].astype(float).values, requires_grad=True).float().unsqueeze(1).to(device)
        y_star = torch.tensor(df_test['y'].astype(float).values, requires_grad=True).float().unsqueeze(1).to(device)

        inputs = torch.cat([x_star, y_star], dim=1)

        u_star = self.u_net(inputs)
        v_star = self.v_net(inputs)
        p_star = self.p_net(inputs)

        return u_star.cpu().detach().numpy(), v_star.cpu().detach().numpy(), p_star.cpu().detach().numpy()