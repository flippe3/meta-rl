import gym
import metaworld
import random
from gym.wrappers import TimeLimit
from sb3_contrib import TRPO
from stable_baselines3.common.env_util import make_vec_env
import os, cv2

# Set variables
resolution = (1920,1080)
camera = "corner" # Camera is one of ['corner', 'topview', 'behindGripper', 'gripperPOV']
flip = False

for t_name,_ in metaworld.ML1.ENV_NAMES:
    ml1 = metaworld.ML1(t_name)
    env = ml1.train_classes[t_name]()
    task = random.choice(ml1.train_tasks)
    env.set_task(task)
    env = TimeLimit(env, max_episode_steps=500)

    model = TRPO("MlpPolicy", env, verbose=1)
    model.load(t_name+"_TRPO_1M.zip")

    def writer_for(tag, fps, res):
        if not os.path.exists('trpo_ml10'):
            os.mkdir('trpo_ml10')
        return cv2.VideoWriter(
            f'trpo_ml10/{tag}.avi',
            cv2.VideoWriter_fourcc('m','j','p','g'),
            fps,
            res)

    def trajectory_generator(env, res=(640, 480), camera='corner'):
        o = env.reset()
        for _ in range(env.max_path_length):
            a, agent_info = model.predict(o)
            o, r, done, info = env.step(a)
            # camera is one of ['corner', 'topview', 'behindgripper', 'gripperpov']
            yield r, done, info, env.sim.render(*res, mode='offscreen', camera_name=camera)[:,:,::-1]
            
    def run_task(env, name):
        print(f"Recording: {name}")
        writer = writer_for(name, env.metadata['video.frames_per_second'], resolution)
        for r, done, info, img in trajectory_generator(env, resolution, camera):
            if flip: 
                img = cv2.rotate(img, cv2.ROTATE_180)
            writer.write(img)
        writer.release()
        env.close()

    run_task(env, t_name) 
