# ==============================================
# CDS524 Assignment 1 - Q-Learning Game (Training Only)
# 精简版：移除手动模式按钮/说明 + 保留训练核心功能 + 零卡顿
# 依赖：pip install numpy pygame
# 操作：T=Train R=Reset Q=Quit
# ==============================================
import numpy as np
import random
import sys
import time
import pygame

# -------------------------- 全局配置 --------------------------
# 网格配置
GRID_SIZE = 5
START = (0, 0)
TARGET = (4, 4)
ACTION_MAP = {0: 'UP', 1: 'DOWN', 2: 'LEFT', 3: 'RIGHT'}
# UI配置（方格下移，无重叠）
UI_WIDTH, UI_HEIGHT = 800, 700
CELL_SIZE = 80
UI_X_OFFSET = (UI_WIDTH - GRID_SIZE * CELL_SIZE) // 2
UI_Y_OFFSET = 100  # 方格下移，避开顶部文字
# 颜色配置
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
BLUE = (0, 0, 255)    # 起点
GOLD = (255, 215, 0)  # 终点
CYAN = (0, 255, 255)  # 智能体
GRAY = (200, 200, 200)# 路径
GREEN = (0, 255, 0)   # 按钮
RED = (255, 0, 0)     # 退出按钮
# 全局状态
QUIT_FLAG = False
MODE = "IDLE"  # IDLE/TRAIN
LAST_OP_TIME = 0  # 操作防抖

# -------------------------- 1. 网格环境类 --------------------------
class GridEnv:
    def __init__(self):
        self.cur_pos = START
        self.reset()

    def reset(self):
        self.cur_pos = START
        self.path = [START]
        return self.cur_pos

    def is_valid(self, x, y):
        return 0 <= x < GRID_SIZE and 0 <= y < GRID_SIZE

    def step(self, action):
        x, y = self.cur_pos
        if action == 0: x -= 1  # UP
        elif action == 1: x += 1# DOWN
        elif action == 2: y -= 1# LEFT
        elif action == 3: y += 1# RIGHT

        pre_pos = self.cur_pos
        if self.is_valid(x, y):
            self.cur_pos = (x, y)
            self.path.append(self.cur_pos)
        done = self.cur_pos == TARGET
        reward = self._get_reward(pre_pos, self.cur_pos, done)
        return self.cur_pos, reward, done

    def _get_reward(self, pre_pos, cur_pos, done):
        if done: return 100    # Reach target reward
        if pre_pos == cur_pos: return -10 # Invalid move penalty
        return -1               # Normal move penalty

# -------------------------- 2. Q-Learning智能体类 --------------------------
class QLAgent:
    def __init__(self):
        self.state_n = GRID_SIZE * GRID_SIZE
        self.action_n = 4
        self.alpha = 0.1      # Learning rate
        self.gamma = 0.9      # Discount factor
        self.epsilon = 1.0    # Exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.Q_table = np.zeros((self.state_n, self.action_n))
        self.train_episodes = 0
        self.step = 0
        self.total_reward = 0

    def pos2state(self, pos):
        x, y = pos
        return x * GRID_SIZE + y

    def choose_action(self, pos):
        state = self.pos2state(pos)
        if random.uniform(0, 1) < self.epsilon:
            return random.randint(0, 3)  # Exploration
        else:
            return np.argmax(self.Q_table[state])  # Exploitation

    def update_q(self, pre_pos, action, reward, cur_pos, done):
        s = self.pos2state(pre_pos)
        s_ = self.pos2state(cur_pos)
        target = reward if done else reward + self.gamma * np.max(self.Q_table[s_])
        self.Q_table[s, action] += self.alpha * (target - self.Q_table[s, action])
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def save_q(self):
        """Save Q-table to local"""
        try:
            np.save("training_only_q_table.npy", self.Q_table)
            print("\n✅ Q-table saved: training_only_q_table.npy")
        except:
            print("\n⚠️  Failed to save Q-table")

    def reset_stats(self):
        """Reset training statistics"""
        self.step = 0
        self.total_reward = 0

# -------------------------- 3. Pygame UI类（移除手动模式）--------------------------
class GameUI:
    def __init__(self, env, agent):
        pygame.init()
        self.screen = pygame.display.set_mode((UI_WIDTH, UI_HEIGHT))
        pygame.display.set_caption("CDS524 Q-Learning Grid (Training Only)")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("Arial", 22)
        self.small_font = pygame.font.SysFont("Arial", 18)
        self.env = env
        self.agent = agent
        # 移除Manual按钮，仅保留Train/Reset/Quit
        self.buttons = {
            "train": (180, 550, 150, 50, GREEN, "Train"),
            "reset": (380, 550, 150, 50, GREEN, "Reset"),
            "quit": (600, 550, 100, 50, RED, "Quit")
        }

    def draw_grid(self):
        """Draw grid UI"""
        self.screen.fill(WHITE)
        for x in range(GRID_SIZE):
            for y in range(GRID_SIZE):
                rect_x = UI_X_OFFSET + y * CELL_SIZE
                rect_y = UI_Y_OFFSET + x * CELL_SIZE
                cell_rect = pygame.Rect(rect_x, rect_y, CELL_SIZE, CELL_SIZE)
                # Draw cell background
                if (x, y) == START:
                    pygame.draw.rect(self.screen, BLUE, cell_rect)
                elif (x, y) == TARGET:
                    pygame.draw.rect(self.screen, GOLD, cell_rect)
                elif (x, y) in self.env.path[:-1]:
                    pygame.draw.rect(self.screen, GRAY, cell_rect)
                # Draw cell border
                pygame.draw.rect(self.screen, BLACK, cell_rect, 2)
                # Draw agent
                if (x, y) == self.env.cur_pos:
                    agent_rect = pygame.Rect(rect_x + 10, rect_y + 10, CELL_SIZE - 20, CELL_SIZE - 20)
                    pygame.draw.rect(self.screen, CYAN, agent_rect)

    def draw_buttons(self):
        """Draw operation buttons (no Manual button)"""
        for btn_name, (x, y, w, h, color, text) in self.buttons.items():
            pygame.draw.rect(self.screen, color, (x, y, w, h))
            pygame.draw.rect(self.screen, BLACK, (x, y, w, h), 2)
            text_surf = self.small_font.render(text, True, BLACK)
            text_rect = text_surf.get_rect(center=(x + w//2, y + h//2))
            self.screen.blit(text_surf, text_rect)

    def draw_info(self):
        """Draw info panel (remove all manual mode instructions)"""
        # Top info
        info_y = 10
        mode_text = self.font.render(f"Current Mode: {MODE}", True, BLACK)
        self.screen.blit(mode_text, (50, info_y))
        
        # Second line info
        info_y += 35
        pos_text = self.font.render(f"Agent Position: {self.env.cur_pos}", True, BLACK)
        self.screen.blit(pos_text, (50, info_y))
        step_text = self.font.render(f"Step Count: {self.agent.step}", True, BLACK)
        self.screen.blit(step_text, (350, info_y))
        reward_text = self.font.render(f"Total Reward: {self.agent.total_reward:.1f}", True, BLACK)
        self.screen.blit(reward_text, (600, info_y))
        
        # Training info (only)
        info_y = 500
        if MODE == "TRAIN":
            eps_text = self.small_font.render(f"Exploration Rate (ε): {self.agent.epsilon:.3f}", True, BLACK)
            self.screen.blit(eps_text, (50, info_y))
            episode_text = self.small_font.render(f"Training Episodes: {self.agent.train_episodes}", True, BLACK)
            self.screen.blit(episode_text, (350, info_y))
        # 完全移除手动模式说明行

    def check_button_click(self, mouse_pos):
        """Detect button click (no Manual button)"""
        for btn_name, (x, y, w, h, _, _) in self.buttons.items():
            if x <= mouse_pos[0] <= x + w and y <= mouse_pos[1] <= y + h:
                return btn_name
        return None

    def update(self):
        """Update UI"""
        self.draw_grid()
        self.draw_buttons()
        self.draw_info()
        pygame.display.update()
        self.clock.tick(60)  # Smooth training visualization

# -------------------------- 4. 按键检测（移除手动模式相关）--------------------------
def handle_keyboard(agent, env):
    """Keyboard handling (training only)"""
    global MODE, QUIT_FLAG, LAST_OP_TIME
    current_time = time.time()
    
    # Debounce: prevent multiple operations
    if current_time - LAST_OP_TIME < 0.2:
        return
    
    # Get all key states
    keys = pygame.key.get_pressed()
    
    # Quit (highest priority)
    if keys[pygame.K_q]:
        QUIT_FLAG = True
        agent.save_q()
        print("👋 Quit pressed, exiting game...")
        LAST_OP_TIME = current_time
        return
    
    # Reset
    if keys[pygame.K_r]:
        env.reset()
        agent.reset_stats()
        agent.epsilon = 1.0
        MODE = "IDLE"
        print("🔄 Game reset! Agent back to start position")
        LAST_OP_TIME = current_time
        return
    
    # Switch to Train mode (only)
    if keys[pygame.K_t] and MODE != "TRAIN":
        MODE = "TRAIN"
        print("🚀 Switched to Training Mode")
        LAST_OP_TIME = current_time
        return

# -------------------------- 5. 训练模式函数 --------------------------
def run_train_step(agent, env):
    """Training step with stable logic"""
    pre_pos = env.cur_pos
    action = agent.choose_action(pre_pos)
    cur_pos, reward, done = env.step(action)
    
    agent.step += 1
    agent.total_reward += reward
    agent.update_q(pre_pos, action, reward, cur_pos, done)
    
    # Reset when reach target
    if done:
        agent.train_episodes += 1
        print(f"🏆 Training Episode {agent.train_episodes} | Steps: {agent.step} | Reward: {agent.total_reward:.1f}")
        env.reset()
        agent.reset_stats()
    time.sleep(0.05)  # Visible training steps

# -------------------------- 主逻辑 --------------------------
def main():
    global MODE, QUIT_FLAG
    # Initialize
    env = GridEnv()
    agent = QLAgent()
    ui = GameUI(env, agent)
    
    # Welcome info (training only)
    print("="*60)
    print("🎉 CDS524 Q-Learning Game (Training Only Version)")
    print("✅ Removed: Manual Mode (buttons & instructions)")
    print("📋 Controls:")
    print("   T = Train Mode | R = Reset | Q = Quit")
    print("="*60)
    
    # Main loop
    while not QUIT_FLAG:
        # Handle Pygame events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                QUIT_FLAG = True
                agent.save_q()
            # Mouse button click (no Manual button)
            elif event.type == pygame.MOUSEBUTTONDOWN:
                btn = ui.check_button_click(pygame.mouse.get_pos())
                if btn == "train":
                    MODE = "TRAIN"
                    print("🚀 Train button clicked - Training Mode")
                elif btn == "reset":
                    env.reset()
                    agent.reset_stats()
                    agent.epsilon = 1.0
                    MODE = "IDLE"
                    print("🔄 Reset button clicked - Game reset")
                elif btn == "quit":
                    QUIT_FLAG = True
                    agent.save_q()
        
        # Handle keyboard input (training only)
        handle_keyboard(agent, env)
        
        # Run training if in TRAIN mode
        if MODE == "TRAIN" and not QUIT_FLAG:
            run_train_step(agent, env)
        
        # Update UI
        ui.update()
    
    # Cleanup
    pygame.quit()
    print("\n👋 Game exited successfully!")
    sys.exit(0)

# -------------------------- 运行入口 --------------------------
if __name__ == "__main__":
    # Check dependencies
    try:
        import numpy, pygame
    except ImportError as e:
        print(f"❌ Missing dependency: {e}")
        print("👉 Install required packages: pip install numpy pygame")
        input("\nPress Enter to exit...")
        sys.exit(1)
    
    # Run with error handling
    try:
        main()
    except Exception as e:
        print(f"\n❌ Runtime error: {str(e)}")
        input("Press Enter to exit...")
        sys.exit(1)