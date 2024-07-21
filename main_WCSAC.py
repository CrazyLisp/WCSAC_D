import torch 
import argparse

def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    # --ALGORITHM PARAMETERS--
    parser.add_argument("--algo_sim4exp",   type=str,       default="WCSAC")
    parser.add_argument("--n-episode",      type=int,       default=2000)
    
    parser.add_argument("--actor-lr",       type=float,     default=1e-4)
    parser.add_argument("--actor-betas", type=list, default=[0.9, 0.999])
    parser.add_argument("--critic-lr",      type=float,     default=1e-3)
    parser.add_argument("--critic-betas", type=list, default=[0.9, 0.999])
    parser.add_argument("--alpha-lr",       type=float,     default=1e-4)
    parser.add_argument("--alpha-betas", type=list, default=[0.9, 0.999])
    parser.add_argument("--beta-lr",       type=float,     default=1e-4)
    parser.add_argument("--hidden-dim",     type=int,       default=256)
    parser.add_argument("--gamma",          type=float,     default=0.98)
    parser.add_argument("--tau",            type=float,     default=0.005)
    parser.add_argument("--risk_level",            type=int,     default=0.5)
    parser.add_argument("--T",            type=int,     default=50)
    parser.add_argument("--cost-limit",    type=int,     default=20)
    parser.add_argument("--damp-scale",    type=int,     default=10)
    
    parser.add_argument("--buffer-size",    type=float,     default=100000)
    parser.add_argument("--minimal-size",   type=float,     default=1000)
    parser.add_argument("--batch-size",     type=float,     default=256)
    
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0" if torch.cuda.is_available() else "cpu",
    )
    return parser.parse_known_args()[0]

from wcsac_pytorch import WCSACAgent, ReplayBuffer
import safety_gymnasium
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
def train_WCSAC(args: argparse.Namespace = get_args(), **kwargs) -> None:
    env_id = 'SafetyPointGoal1-v0'
    env = safety_gymnasium.make(env_id)
    args.state_dim = env.observation_space.shape[0]
    args.action_dim = 25
    SET_ACTION_D = [-1, -0.5, 0, 0.5, 1]
    args.target_entropy = -1
    agent = WCSACAgent(args.state_dim,
                       args.hidden_dim,
                       args.action_dim,
                       args.actor_lr,
                       args.actor_betas,
                       args.critic_lr,
                       args.critic_betas,
                       args.alpha_lr,
                       args.alpha_betas,
                       args.beta_lr,
                       args.target_entropy,
                       args.tau,
                       args.gamma,
                       args.cost_limit,
                       args.T,
                       args.risk_level,
                       args.damp_scale,
                       args.device)
    
    replay_buffer = ReplayBuffer(args.buffer_size)
    
    return_list = []
    cost_list   = []
    with tqdm(total= args.n_episode) as pbar:
        for i_episode in range(args.n_episode):
            episode_return = 0
            episode_cost   = 0
            state, _ = env.reset()
            done = False
            while not done:
                action = agent.take_action(state)
                a_1 = SET_ACTION_D[int(action%5)]
                a_2 = SET_ACTION_D[int(np.floor(action/5))]
                joint_action = [a_1, a_2]
                next_state, reward, cost, terminated, truncated, _ = env.step(joint_action)
                if terminated or truncated:
                    done = True
                # exit()
                replay_buffer.add(state, action, reward, cost, next_state, done)
                state = next_state
                episode_return += reward
                episode_cost   += cost
                if replay_buffer.size() > args.minimal_size:
                    b_s, b_a, b_r, b_c,  b_ns, b_d = replay_buffer.sample(args.batch_size)
                    transition_dict = {'states': b_s, 'actions': b_a, 'next_states': b_ns, 'rewards': b_r, "costs":b_c,
                                        'dones': b_d}
                    agent.update(transition_dict)

            return_list.append(episode_return)
            cost_list.append(episode_cost)
            pbar.update(1)
            
            if (i_episode + 1) % 10 == 0:
                episodes_list = list(range(len(return_list)))
                plt.figure(figsize=(8,6))
                plt.plot(episodes_list, return_list)
                plt.xlabel("Episodes")
                plt.ylabel("Returns")
                plt.title("WCSAC TEST DEMO")
                plt.savefig("return.png")

                episodes_list = list(range(len(cost_list)))
                plt.figure(figsize=(8,6))
                plt.plot(episodes_list, cost_list)
                plt.xlabel("Episodes")
                plt.ylabel("Returns")
                plt.title("WCSAC TEST DEMO")
                plt.savefig("cost.png")
    
if __name__ == "__main__":
    

    train_WCSAC()
