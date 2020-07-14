
import tensorflow as tf
from tensorflow_privacy.privacy.analysis.rdp_accountant import compute_rdp
from tensorflow_privacy.privacy.analysis.rdp_accountant import get_privacy_spent
import math

'''
def calculateEps(p,q):
    numerator=(p+(1-p)*q)
    denominator=(1-p)*q
    eps=math.log(numerator/denominator)
    return eps
#p=[.9,.9,.9,.9,.9]
#q=[0.003,0.022,0.16,.9]
#p=[.9,.9,.9,.9,.9,.9,.9,.9,.9]
#q=[.1,0.2,.3,0.4,.5,0.6,.7,.8,.9]
p=[.1,.3,.6,.8,.8,.9]
q=[.8,.6,.8,.6,.2,.1]
for i,j in zip(p,q):
    eps = math.log((i + (1 - i) * j) / ((1 - i) * j))
    print(eps)
    #print("for p=% and q=%",(i,j))
    #print(calculateEps(i,j))
'''
batch=500
microbatch=20
learning_rate=0.25
epoch=50
size=60000
delta=0.000001

def compute_epsilon(steps,noise):
  """Computes epsilon value for given hyperparameters."""
  if noise == 0.0:
    return float('inf')
  orders = [1 + x / 10. for x in range(1, 100)] + list(range(12, 64))
  sampling_probability = batch / size
  rdp = compute_rdp(q=sampling_probability,
                    noise_multiplier=noise,
                    steps=steps,
                    orders=orders)
  # Delta is set to 1e-5 because MNIST has 60000 training points.
  return get_privacy_spent(orders, rdp, target_delta=delta)[0]

for i in [30,3.5,1.90,1.15,.920,.810,.743]:
    eps = compute_epsilon(epoch * size // batch, i)
    print('For delta=%4f, the current epsilon is: %.4f' % (delta, eps))