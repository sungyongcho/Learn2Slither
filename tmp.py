from typing import List, Tuple, Union  # Make sure these are imported

import numpy as np  # If using numpy arrays for actions

# Assume constants are defined elsewhere:
# REWARD_LIVING_STEP = ...
# REWARD_RED_APPLE = ...
# REWARD_GREEN_APPLE = ...
# REWARD_DEATH = ...
# STARVE_FACTOR = ... # e.g., 100


class Environment:  # Your Environment class
    # ... (other methods like __init__, _move, _place_apples, is_collision, etc.)
    # In __init__, you'd likely set self.step_by_step, e.g.:
    # def __init__(self, ..., step_by_step: bool = False):
    #     self.step_by_step = step_by_step
    #     self.frame = 0
    #     self.frames_since_food = 0
    #     # ... other initializations (snake, apples, head, width, height)

    def _get_action_name(
        self, action_vec: Union[np.ndarray, List[int]]
    ) -> str:  # Helper method
        if isinstance(action_vec, np.ndarray):
            action_vec = action_vec.tolist()
        if action_vec == [1, 0, 0]:
            return "straight"
        if action_vec == [0, 1, 0]:
            return "right_turn"
        if action_vec == [0, 0, 1]:
            return "left_turn"
        return str(action_vec)

    def step(
        self,
        action: Union[np.ndarray, List[int]],
    ) -> Tuple[int, bool, int]:
        """Advance one tick. Return (learning_reward, game_over, display_score)."""

        action_name = self._get_action_name(action)  # Get readable action name

        if self.step_by_step:
            print(
                f"[Env Step] Frame: {self.frame + 1}, Frames since food: {self.frames_since_food + 1}, Received action: {action_name} ({action})"
            )
            print(
                f"[Env Step] Current snake head: {self.head}, Snake length: {len(self.snake)}"
            )
            # To print the whole snake: print(f"[Env Step] Current snake: {self.snake}")

        self.frame += 1
        self.frames_since_food += 1

        # Store old head position for logging if needed, though self.head is updated by _move
        # old_head_pos = self.head # If self.head is a mutable object, ensure deep copy if logging old vs new from same object

        # 1. Move head
        self._move(action)  # This method updates self.head
        self.snake.insert(0, self.head)  # new head at index 0

        if self.step_by_step:
            print(
                f"[Env Step] Movement: Action '{action_name}' led to new head position: {self.head}. Snake length before pop/growth: {len(self.snake)}"
            )

        grew = shrink = False
        reward = REWARD_LIVING_STEP
        if self.step_by_step:
            print(f"[Env Step] Initial reward for step: {reward}")

        # 2. Apple checks ────────────────
        eaten_red_apple_pos = None
        eaten_green_apple_pos = None

        if self.red_apple and self.head == self.red_apple:
            eaten_red_apple_pos = self.red_apple
            reward = REWARD_RED_APPLE
            self.frames_since_food = 0
            shrink = True
            if self.step_by_step:
                print(
                    f"[Env Step] Event: Ate RED apple at {eaten_red_apple_pos}. Reward: {reward}. Frames since food reset. Flagged to shrink."
                )
            self._place_red_apple()  # This should place a new one
            if self.step_by_step:
                print(
                    f"[Env Step] New red apple placed at: {self.red_apple if self.red_apple else 'None (no space?)'}"
                )

        elif self.head in self.green_apples:
            eaten_green_apple_pos = self.head  # The head is on the apple
            self.green_apples.remove(self.head)
            reward = REWARD_GREEN_APPLE
            self.frames_since_food = 0
            grew = True
            if self.step_by_step:
                print(
                    f"[Env Step] Event: Ate GREEN apple at {eaten_green_apple_pos}. Reward: {reward}. Frames since food reset. Flagged to grow."
                )
            self._place_green_apples()  # This should attempt to maintain apple count
            if self.step_by_step:
                print(
                    f"[Env Step] Green apples updated. Current green apples: {len(self.green_apples)} positions: {self.green_apples}"
                )

        # 3. Update length ──────────────
        # Length is len(self.snake) after head insertion.
        # grew=True means length increases by 1 from original.
        # normal means length stays same as original.
        # shrink=True means length decreases by 1 from original.

        if (
            not grew and self.snake
        ):  # Normal move (not grown) or red apple (will shrink more)
            popped_item = self.snake.pop()  # normal move → drop tail
            if self.step_by_step:
                print(
                    f"[Env Step] Length update: Not grown, so tail popped ({popped_item}). Snake length now: {len(self.snake)}"
                )
        elif grew and self.step_by_step:
            print(
                f"[Env Step] Length update: Grown (green apple), tail not popped. Snake length now: {len(self.snake)}"
            )

        if shrink and self.snake:  # Red apple effect
            popped_item_shrink = self.snake.pop()  # red apple → shrink extra
            if self.step_by_step:
                print(
                    f"[Env Step] Length update: Shrunk (red apple), additional tail popped ({popped_item_shrink}). Snake length now: {len(self.snake)}"
                )
            if not self.snake:  # shrunk away completely
                if self.step_by_step:
                    print(
                        f"[Env Step] GAME OVER: Snake shrunk to nothing. Final Reward: {REWARD_DEATH}, Final Score: {len(self.snake)}"
                    )
                return REWARD_DEATH, True, len(self.snake)  # Score is 0

        # 4. Wall / body collision (after tail possibly removed)
        if (
            self.is_collision()
        ):  # is_collision should check self.head against walls and self.snake[1:]
            if self.step_by_step:
                # You might want to add more detail in is_collision itself or get detail from it
                print(
                    f"[Env Step] GAME OVER: Collision detected at head position {self.head}. Snake: {self.snake}. Final Reward: {REWARD_DEATH}, Final Score: {len(self.snake)}"
                )
            return REWARD_DEATH, True, len(self.snake)

        # 5. Starvation check (uses final length for this step)
        starvation_limit = (
            STARVE_FACTOR * len(self.snake) if self.snake else STARVE_FACTOR
        )  # Handle len(self.snake)=0 case for limit
        if self.frames_since_food > starvation_limit:
            if self.step_by_step:
                print(
                    f"[Env Step] GAME OVER: Starvation. Frames since food ({self.frames_since_food}) > limit ({STARVE_FACTOR} * {len(self.snake)} = {starvation_limit}). Final Reward: {REWARD_DEATH}, Final Score: {len(self.snake)}"
                )
            return REWARD_DEATH, True, len(self.snake)

        if self.step_by_step:
            print(
                f"[Env Step] Step finished successfully. Reward: {reward}, Game Over: False, Score: {len(self.snake)}"
            )
        return reward, False, len(self.snake)
