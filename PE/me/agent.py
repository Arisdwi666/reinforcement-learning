import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Importing function from the env.py
from env import final_states


# Membuat kelas untuk tabel Q-learning
class QLearningTable:
    def __init__(self, actions, learning_rate=0.05, reward_decay=0.99, e_greedy=0.99):
        
        self.actions = actions # List of actions
        self.lr = learning_rate # Learning rate
        self.gamma = reward_decay # Value of gamma
        self.epsilon = e_greedy # Value of epsilon
        
        # Membuat tabel Q lengkap untuk semua sel
        self.q_table = pd.DataFrame(columns=self.actions, dtype=np.float64)
        # Membuat tabel-Q untuk sel-sel dari rute terakhir
        self.q_table_final = pd.DataFrame(columns=self.actions, dtype=np.float64)


    # Fungsi untuk memilih tindakan untuk agen
    def choose_action(self, observation):
        
        # Memeriksa apakah state ada di tabel
        self.check_state_exist(observation)
        
        # Pemilihan aksi - 90% sesuai dengan epsilon == 0.9
        # Memilih tindakan terbaik
        if np.random.uniform() < self.epsilon:
            state_action = self.q_table.loc[observation, :]
            state_action = state_action.reindex(np.random.permutation(state_action.index))
            action = state_action.idxmax()
        else:
            # Memilih tindakan acak - tersisa 10% untuk memilih secara acak
            action = np.random.choice(self.actions)
        return action


    # Berfungsi untuk mempelajari dan memperbarui tabel Q dengan pengetahuan baru
    def learn(self, state, action, reward, next_state):
        # Memeriksa apakah langkah selanjutnya ada di tabel-Q
        self.check_state_exist(next_state)

        # Keadaan saat ini di posisi saat ini
        q_predict = self.q_table.loc[state, action]

        # Memeriksa apakah keadaan berikutnya bebas atau hambatan atau tujuan
        if next_state != 'goal' or next_state != 'obstacle':
            q_target = reward + self.gamma * self.q_table.loc[next_state, :].max()
        else:
            q_target = reward

        # Memperbarui tabel-Q dengan pengetahuan baru
        self.q_table.loc[state, action] += self.lr * (q_target - q_predict)

        return self.q_table.loc[state, action]

    # Menambah status baru Q-tabel
    def check_state_exist(self, state):
        if state not in self.q_table.index:
            self.q_table = self.q_table.append(
                pd.Series(
                    [0]*len(self.actions),
                    index=self.q_table.columns,
                    name=state,
                )
            )

    # Mencetak tabel-Q dengan status
    def print_q_table(self):
        # Mendapatkan koordinat rute akhir dari env.py
        e = final_states()

        # Membandingkan indeks dengan koordinat dan menulis nilai tabel Q baru
        for i in range(len(e)):
            state = str(e[i])  # state = '[5.0, 40.0]'
            # Memeriksa semua indeks dan memeriksa
            for j in range(len(self.q_table.index)):
                if self.q_table.index[j] == state:
                    self.q_table_final.loc[state, :] = self.q_table.loc[state, :]

        print()
        print('Panjang Q-table final=', len(self.q_table_final.index))
        print('Final Q-table dengan nilai dari rute akhir:')
        print(self.q_table_final)

        print()
        print('Length Q-table penuh =', len(self.q_table.index))
        print('Full Q-table:')
        print(self.q_table)

    # Plotting the results for the number of steps
    def plot_results(self, steps, cost):
        #
        f, (ax1, ax2) = plt.subplots(nrows=1, ncols=2)
        #
        ax1.plot(np.arange(len(steps)), steps, 'b')
        ax1.set_xlabel('Episode')
        ax1.set_ylabel('Steps')
        ax1.set_title('Episode via steps')

        #
        ax2.plot(np.arange(len(cost)), cost, 'r')
        ax2.set_xlabel('Episode')
        ax2.set_ylabel('Cost')
        ax2.set_title('Episode via cost')

        plt.tight_layout()  # Function to make distance between figures

        #
        plt.figure()
        plt.plot(np.arange(len(steps)), steps, 'b')
        plt.title('Episode via steps')
        plt.xlabel('Episode')
        plt.ylabel('Steps')

        #
        plt.figure()
        plt.plot(np.arange(len(cost)), cost, 'r')
        plt.title('Episode via cost')
        plt.xlabel('Episode')
        plt.ylabel('Cost')

        # Showing the plots
        plt.show()
