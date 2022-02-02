# Image-Dehazing using Contextual Regularization 

Image dehazing is a well-known ill-posed problem, which usually requires some image priors to make the problem well-posed. Single image dehazing aims to estimate a haze-free image
from a hazy image. It is a classical image processing problem, which has been an active research topic in the vision and graphics communities within the last decade. 

## Method
An efficient regularization method to remove haze from a single input image. This method benefits much from an exploration of the inherent boundary constraint on the transmission
function.
This constraint, combined with a weighted L1−norm based contextual regularization, is modeled into an optimization problem to estimate the unknown scene transmission.
A quite efficient algorithm based on variable splitting is also presented to solve the problem. This method requires only a few general assumptions and can restore a high-quality haze-free image with faithful colors and fine image details.

- Estimating Global AirlightCalculating Boundary constraints
- Refine Estimation
- Imaging Model And Problem Constraints
- Boundary Constraint from Radiance Cube
- Weighted L1-norm based Contextual Regularization
- Scene Transmission Estimation
- A bank of high-order filters used.It consists of eight Kirsch operators and a Laplacian operator for preserving image edges and corners.

---

## GROUP MEMBERS 

| NAME  | ROLL NUMBER |
| ------------- | ------------- |
| ABHIJITH A THAMPI | AM.EN.U4AIE20102  |
| ADITHYAN M NAIR  | AM.EN.U4AIE20105   |
| AJAY G NAIR  | AM.EN.U4AIE20108  |
| DEVAKRISHNA SANILKUMAR | AM.EN.U4AIE20119 |
| GOVIND NANDAKUMAR | AM.EN.U4AIE20129 |

Dataset Link: https://drive.google.com/drive/folders/19yXNAaPBHbQmEU563AZ5w7Rihpz13p_N
