import pygame
import math
import numpy as np

# Constants
WIDTH, HEIGHT = 600, 600
BACKGROUND_COLOR = (30, 30, 30)
HEXAGON_COLOR = (0, 255, 0)
BALL_COLOR = (255, 0, 0)
GRAVITY = 0.2
BOUNCE_DAMPENING = 0.85
ROTATION_SPEED = 1  # Degrees per frame

# Initialize Pygame
pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
clock = pygame.time.Clock()

# Hexagon parameters
hex_radius = 150
hex_center = np.array([WIDTH // 2, HEIGHT // 2])
angle = 0  # Rotation angle

# Ball parameters
ball_pos = np.array([WIDTH // 2, HEIGHT // 2 - hex_radius + 10], dtype=float)
ball_velocity = np.array([2, 0], dtype=float)
ball_radius = 10

def get_hexagon_points(center, radius, rotation):
    """Returns the hexagon vertex points rotated by 'rotation' degrees."""
    points = []
    for i in range(6):
        angle = math.radians(i * 60 + rotation)
        x = center[0] + radius * math.cos(angle)
        y = center[1] + radius * math.sin(angle)
        points.append((x, y))
    return points

def reflect_vector(velocity, normal):
    """Reflect a velocity vector off a given normal."""
    return velocity - 2 * np.dot(velocity, normal) * normal

def update_ball():
    global ball_pos, ball_velocity
    
    # Apply gravity
    ball_velocity[1] += GRAVITY
    ball_pos += ball_velocity
    
    # Get hexagon edges
    hex_points = get_hexagon_points(hex_center, hex_radius, angle)
    for i in range(6):
        p1 = np.array(hex_points[i])
        p2 = np.array(hex_points[(i + 1) % 6])
        edge_vector = p2 - p1
        edge_normal = np.array([-edge_vector[1], edge_vector[0]])
        edge_normal /= np.linalg.norm(edge_normal)
        
        # Check if ball crosses the edge
        to_ball = ball_pos - p1
        distance = np.dot(to_ball, edge_normal)
        
        if distance < ball_radius:
            # Reflect velocity and move ball out of collision
            ball_velocity = reflect_vector(ball_velocity, edge_normal) * BOUNCE_DAMPENING
            ball_pos += edge_normal * (ball_radius - distance)

# Main loop
running = True
while running:
    screen.fill(BACKGROUND_COLOR)
    
    # Handle events
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
    
    # Update hexagon rotation
    angle += ROTATION_SPEED
    
    # Update ball movement
    update_ball()
    
    # Draw hexagon
    hex_points = get_hexagon_points(hex_center, hex_radius, angle)
    pygame.draw.polygon(screen, HEXAGON_COLOR, hex_points, 2)
    
    # Draw ball
    pygame.draw.circle(screen, BALL_COLOR, ball_pos.astype(int), ball_radius)
    
    pygame.display.flip()
    clock.tick(60)

pygame.quit()
