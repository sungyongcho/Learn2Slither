## Learn2Slither

### TODO
- **fix the state**
```
Distance Measurements (16 neurons)
Each direction contains 4 normalized distance values representing:

Distance to the nearest wall
Distance to the nearest green apple
Distance to the nearest red apple
Distance to the nearest snake body segment
Danger Detection (4 neurons)
Four binary neurons (0 or 1) detect immediate collision threats in adjacent cells. These neurons activate when:

A wall is directly adjacent in that direction
A snake body segment is directly adjacent in that direction

```
- fix rewards
- exploitation without learning - no training, just play with current model
- save/load model
```
You must have at least 3 saved model files, trained respectively with 1, 10, and 100
training sessions. This should show how much your snake "learns" over multiple training
sessions. You must be able to start a new session using one of your saved models, in conjuction with the “non-learning” feature, to verify the performance of each model. This
will be tested during the defence.
```
- step by step
    - show the details of the training process i guess?
- args
    - sessions
    - save (save model)
    - load (load model)
    - visual (visualize on off)
    - dontlearn ( no training just play with current model)
    - step-by-step (step by step mode)

