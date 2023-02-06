import gym

env = gym.make('ALE/Breakout-v5',
    obs_type='rgb',                   # ram | rgb | grayscale
    frameskip=4,                      # frame skip
    mode=None,                        # game mode, see Machado et al. 2018
    difficulty=None,                  # game difficulty, see Machado et al. 2018
    repeat_action_probability=0.25,   # Sticky action probability
    full_action_space=False,          # Use all actions
    render_mode=None                  # None | human | rgb_array
)
