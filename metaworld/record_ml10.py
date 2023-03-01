# Might have to run unset LD_PRELOAD to get this to run.
import os, cv2
from garage.experiment import Snapshotter
import metaworld
import random
import time
import gym

# Load the trained model
epoch = 250 # last, first, or integer
snapshotter = Snapshotter()
data = snapshotter.load("/home/fleip/research/garage/metaworld_examples/data/local/experiment/maml_trpo_metaworld_ml10_seed=1_epochs=2000_rollouts_per_task=10_meta_batch_size=20_inner_lr=0.0001_17", itr=epoch)
policy = data['algo'].policy

# Set variables
resolution = (1920,1080)
camera = "corner" # Camera is one of ['corner', 'topview', 'behindGripper', 'gripperPOV']
flip = False

def writer_for(tag, fps, res):
    if not os.path.exists('ml10'):
        os.mkdir('ml10')
    return cv2.VideoWriter(
        f'ml10/{tag}.avi',
        cv2.VideoWriter_fourcc('M','J','P','G'),
        fps,
        res)

def trajectory_generator(env, policy, res=(640, 480), camera='corner'):
    o = env.reset()
    for _ in range(env.max_path_length):
        a, agent_info = policy.get_action(o)
        o, r, done, info = env.step(a)
        # Camera is one of ['corner', 'topview', 'behindGripper', 'gripperPOV']
        yield r, done, info, env.sim.render(*res, mode='offscreen', camera_name=camera)[:,:,::-1]


ml10 = metaworld.ML10() # Construct the benchmark, sampling tasks

# Loading training envs
training_envs = []
training_names = []
for name, env_cls in ml10.train_classes.items():
  env = env_cls()
  task = random.choice([task for task in ml10.train_tasks
                        if task.env_name == name])
  env.set_task(task)
  training_names.append(name)
  training_envs.append(env)

# Load the test tasks
testing_envs = []
test_names = []
for name, env_cls in ml10.test_classes.items():
  env = env_cls()
  task = random.choice([task for task in ml10.test_tasks
                        if task.env_name == name])
  env.set_task(task)
  test_names.append(name)
  testing_envs.append(env)

def run_task(env, name):
    print(f"Recording: {name}")
    writer = writer_for(name, env.metadata['video.frames_per_second'], resolution)
    for r, done, info, img in trajectory_generator(env, policy, resolution, camera):
        if flip: 
            img = cv2.rotate(img, cv2.ROTATE_180)
        writer.write(img)
    writer.release()
    env.close()

print("Running training tasks")
for task in range(len(training_envs)):
    run_task(training_envs[task], f"{training_names[task]}-{epoch}")

print("Running test tasks")
for task in range(len(testing_envs)):
    run_task(testing_envs[task], f"test-{test_names[task]}-{epoch}")


