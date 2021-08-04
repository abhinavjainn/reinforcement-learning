import numpy as np
import tensorflow as tf
from tensorflow import keras
from baselines.common.atari_wrappers import make_atari, wrap_deepmind 
import numpy as np
import gym

seed = 42
model = keras.models.load_model("/content/drive/MyDrive/game_ai/model")

env = make_atari ("BreakoutNoFrameskip-v4")
env = wrap_deepmind(env, frame_stack=True, scale=True)
env.seed(seed)

env = gym.wrappers.Monitor(env,"/content/drive/MyDrive/game_ai/model/videos",video_callable=lambda episode_id:True,force= True)

n_episodes=10
returns=[]

for _ in range(n_episodes):
  ret=0
  state = np.array(env.reset())
  done = False
  while not done:
    # Convert state array to tensor
    state_tensor = tf.convert_to_tensor(state)
    # Add an outer "batch" dimension 
    state_tensor = tf.expand_dims(state_tensor,0)
    # Get action with the highest value from the model
    action = np.argmax(model.predict(state_tensor))
    # Env step using action
    state,reward,done,_ = env.step(action)
    ret+=reward
  returns.append(ret)

env.close()
print("Returns: {}".format(returns))