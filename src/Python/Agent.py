class Agent:
    def __init__(self, x, y, angle, speed=1.0, color=(255, 255, 255), turn_rate=0.1):
        self.x = x
        self.y = y
        self.angle = angle
        self.speed = speed
        self.color = color
        self.turn_rate = turn_rate  # Rate at which the agent can turn