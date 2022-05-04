# AI control of tokamak fusion reactor
- KSTAR is a tokamak (donut-shaped nuclear fusion device) located in South Korea.
- This repository provides an AI that designs the tokamak operation trajectory to control the fusion plasma in KSTAR.
- Here, we would like to control 3 physics parameters; βp, q95 and li.
- I recommend you to see [the tokamak simulator](https://github.com/jaem-seo/KSTAR_tokamak_simulator) first.

# Installation
- You can install by
```
$ git clone https://github.com/jaem-seo/AI_tokamak_control.git
$ cd AI_tokamak_control
```

# 1. Target arrival for 4 s interval
- Open the GUI. It takes a bit depending on your environment.
```
$ python ai_control_v0.py
```
or
```
$ python ai_control_v1.py
```
<p align="center">
  <img src="https://user-images.githubusercontent.com/46472432/166656005-c37156f7-a7a4-4e2c-b714-e0a6319387f7.png">
</p>

- Slide the toggles in the right side to change the target state and press the "AI control" button (it takes a bit).
- Then, the AI will design the tokamak operation trajectory to achieve the target you set.

# 2. Real-time target tracking
- Open the GUI. It takes a bit depending on your environment.
```
$ python rt_control_v0.py
```
or
```
$ python rt_control_v1.py
```
<p align="center">
  <img src="https://user-images.githubusercontent.com/46472432/166666842-0b6ae5d9-1621-4f03-87a0-386ff2931468.png">
</p>

- Slide the toggles in the right side to change the target state.
- Then, the AI will adjust the tokamak operation to track the targets in real-time.

# Note
- The AI was trained by reinforcement learning; [TD3](https://arxiv.org/abs/1802.09477), [HER](https://arxiv.org/abs/1707.01495) implementation from [Stable Baselines](https://github.com/hill-a/stable-baselines).
- The tokamak simulation and the plotting possess most of the computation time, and the AI operation control is very fast.
- Deployment on the KSTAR PCS control system will require further development.

# References
- A. Hill et al. ["Stable Baselines."](https://github.com/hill-a/stable-baselines) GitHub repository (2018).
- S. Fujimoto et al. ["Addressing Function Approximation Error in Actor-Critic Methods."](https://arxiv.org/abs/1802.09477) ICML (2018).
- M. Andrychowicz et al. ["Hindsight Experience Replay."](https://arxiv.org/abs/1707.01495) NIPS (2017).
- J. Seo, ["KSTAR tokamak simulator."](https://github.com/jaem-seo/KSTAR_tokamak_simulator) GitHub repository (2022).
- J. Seo, et al. "Feedforward beta control in the KSTAR tokamak by deep reinforcement learning." Nuclear Fusion [61.10 (2021): 106010.](https://iopscience.iop.org/article/10.1088/1741-4326/ac121b/meta)
- J. Seo, et al. Nuclear Fusion (2022) (In review).
