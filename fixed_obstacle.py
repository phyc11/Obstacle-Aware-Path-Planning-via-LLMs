from openai import OpenAI
import torch
import numpy as np
import time
import pygame
import json
import os
import re
import math

def call_llm(prompt):
    print("Gửi prompt đến LLM:", prompt)
    
    client = OpenAI(
        base_url="https://api.groq.com/openai/v1",
        api_key="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",
    )
    
    completion = client.chat.completions.create(
        model="llama-3.3-70b-versatile",  
        messages=[
            {"role": "system", "content": "You are path planning assistant."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=256,
        temperature=0.2,
        top_p=0.95
    )
    
    response_text = completion.choices[0].message.content
    print("Phản hồi từ LLM:", response_text)
    return response_text

def generate_prompt(agent_pos, goal_pos, obstacles, attempt=1, collision_info=None, current_path=None):
    collision_text = ""
    if collision_info and len(collision_info) > 0 and current_path:
        collision_details = []
        for i, j in collision_info:
            point_i = current_path[i]
            point_j = current_path[j]
            collision_details.append(f"Collision: Between coordinates {point_i} and {point_j} in previous path")
        
        if collision_details:
            collision_text = "Collision information:\n" + "\n".join(collision_details) + "\n\n"
    
    prompt = f"""
    You are a path planning assistant. Given start coordinates, goal coordinates, and obstacles, 
    you need to return a valid path as a JSON object.

    Start: {agent_pos}
    Goal: {goal_pos}
    Obstacles (quadrilaterals): {json.dumps(obstacles)}
    
    IMPORTANT: I need ONLY a JSON object in the format {{"Trajectory": [[x1, y1], [x2, y2], ...]}}
    DO NOT include any code, explanation, or other text. ONLY return the JSON.
    
    This is attempt #{attempt}. Previous attempts failed to generate a collision-free path.
    {collision_text}

    IMPORTANT: DO NOT REPEAT these collisions in the next response.

    The path should:
    1. Start at exactly {agent_pos}
    2. End at exactly {goal_pos}
    3. Avoid all obstacles
    4. Use waypoints as needed to navigate around obstacles safely
    
    Navigation Instructions:
    • Start at the green box, then navigate in 2D plane and
    finally end at the red box.
    • Avoid entering the obstacle areas (orange) at
    any point during the path.
    • The path should include curved motions or additional
    waypoints to smoothly navigate around obstacles and
    restricted zones.

    Obstacle Avoidance Instruction:
    1. Identify which corners of the obstacle areas are at the
    lower side (downside) and which ones are at the up
    per side (upside) based on their y-coordinates. Assume
    that a larger y-coordinate indicates the upper direction.
    2. Identify which corners of the obstacle areas are on the
    left side and which ones are on the right side based on
    their x-coordinates. Assume that a larger x-coordinate
    indicates the right direction.
    3. Assign the following labels to the corners of the obstacle area:
    • Left top corner: C1
    • Right top corner: C2
    • Right bottom corner: C3
    • Left bottom corner: C4
    4. For each corner of the obstacle areas, the path must
    avoid specific directions based on the corner's position. Use the following rules for path planning:
    • For C1 (left top corner): The path must stay either on its left, on its top, or both on its left and top.
    The path (waypoints of path) must never be directly in the region below and the right of C1.
    However,the path can move above and right of C1,or below and left of C1.
    • For C2 (right top corner): The path must stay either on its right, on its top, or both on its right and top.
    The path (waypoints of path) must never be directly in the region below and the left of C2.
    However, the path can move above and left of C2,or below and right of C2.
    • For C3 (right bottom corner): The path must stay
    either on its right, on its bottom, or both on its right
    and bottom. The path (waypoints of path) must never be directly in the region above and the left of C3.
    However, the path can move below and right of C3, or above and left of C3.
    • For C4 (left bottom corner): The path must stay either on its left, on its bottom, or both on its left and
    bottom. The path (waypoints of path) must never be directly in the region above and the right of C4.
    However, the path can move below and left of C4,or above and right of C4.
    5. Ensure that no path points are placed on the edges of
    the obstacle areas. All path points should be located
    entirely outside the obstacle areas, avoiding any points
    along its boundaries.

    Example 1: Starting from the green cirle and ending
    at the red circle while avoiding the obstacle area can
    be done by this trajectory or path:
    
    {{"Trajectory": [[-1.5, 1.2], [-0.8, 0.8], [-0.35, 0.5], [0.0, -0.25], [1.0, -0.5]]}}

    Example 2: Starting from the green circle and ending
    at the red circle while avoiding the obstacle area can
    be done by this trajectory or path:
    
    {{"Trajectory": [[-1.5, 1.2], [-1.5, 0.25], [-1, -0.25], [0.0, -0.35], [1.0, -0.5]]}}

    """
    return prompt

def extract_json_from_response(response):
    """Cố gắng trích xuất JSON từ phản hồi với nhiều phương pháp."""
    json_pattern = r'\{.*"Trajectory"\s*:\s*\[\s*\[.*\]\s*\]\s*\}'
    matches = re.findall(json_pattern, response, re.DOTALL)
    if matches:
        return json.loads(matches[0])
    
    try:
        start_idx = response.find("{")
        end_idx = response.rfind("}")
        if start_idx != -1 and end_idx != -1:
            json_str = response[start_idx:end_idx+1]
            return json.loads(json_str)
    except:
        pass
    try:
        return json.loads(response)
    except:
        print("Không thể trích xuất JSON từ phản hồi")
        return None
    
def line_intersects_polygon(p1, p2, polygon):
    # Kiểm tra xem đoạn thẳng có giao với đa giác không.
    n = len(polygon)
    for i in range(n):
        edge_p1 = polygon[i]
        edge_p2 = polygon[(i + 1) % n]
        
        # Kiểm tra giao điểm của hai đoạn thẳng
        if line_intersects_line(p1, p2, edge_p1, edge_p2):
            return True
    return False

def ccw(a, b, c):
        return (c[1] - a[1]) * (b[0] - a[0]) > (b[1] - a[1]) * (c[0] - a[0])

def line_intersects_line(p1, p2, p3, p4):
    # Kiểm tra xem hai đoạn thẳng có giao nhau không.
    return ccw(p1, p3, p4) != ccw(p2, p3, p4) and ccw(p1, p2, p3) != ccw(p1, p2, p4)

def is_point_inside_polygon(point, polygon):
    # Kiểm tra xem điểm có nằm trong đa giác không
    x, y = point
    inside = False
    n = len(polygon)
    p1x, p1y = polygon[0]
    
    for i in range(n + 1):
        p2x, p2y = polygon[i % n]
        if min(p1y, p2y) < y <= max(p1y, p2y) and x <= max(p1x, p2x):
            if p1y != p2y:
                xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
            if p1x == p2x or x <= xinters:
                inside = not inside
        p1x, p1y = p2x, p2y
    return inside

def check_collision(path, obstacles):
    collision_segments = []
    has_collision = False
    
    for i in range(len(path) - 1):
        segment_has_collision = False
        
        # Kiểm tra từng điểm
        for point in [path[i], path[i+1]]:
            for obs in obstacles:
                if is_point_inside_polygon(point, obs):
                    print(f"Phát hiện va chạm tại điểm {point}")
                    segment_has_collision = True
                    has_collision = True
        
        # Kiểm tra đoạn đường có giao với đa giác không
        for obs in obstacles:
            if line_intersects_polygon(path[i], path[i+1], obs):
                print(f"Phát hiện va chạm trên đoạn từ {path[i]} đến {path[i+1]}")
                segment_has_collision = True
                has_collision = True
        
        if segment_has_collision:
            collision_segments.append((i, i+1))
    
    return has_collision, collision_segments  

def plan_path(agent_pos, goal_pos, obstacles, max_attempts=5):
    all_paths = []  
    all_collision_segments = []  
    
    for attempt in range(max_attempts):
        print(f"Lần thử {attempt+1}/{max_attempts}")
        collision_info = all_collision_segments[-1] if all_collision_segments else None
        current_path = all_paths[-1] if all_paths else None
        prompt = generate_prompt(agent_pos, goal_pos, obstacles, attempt+1, collision_info, current_path)
        response = call_llm(prompt)
        
        try:
            response_json = extract_json_from_response(response)
            if response_json:
                path = response_json.get("Trajectory", [])
                print(f"Đường đi trích xuất: {path}")
                
                if path and all(isinstance(pos, list) and len(pos) == 2 for pos in path):
                    if path[0] != list(agent_pos):
                        path.insert(0, list(agent_pos))
                    if path[-1] != list(goal_pos):
                        path.append(list(goal_pos))
                        
                    has_collision, collision_segments = check_collision(path, obstacles)
                    
                    # Lưu đường đi và các đoạn va chạm
                    all_paths.append(path)
                    all_collision_segments.append(collision_segments)
                    
                    if not has_collision:
                        print("Tìm thấy đường đi hợp lệ:", path)
                        return path, collision_segments, all_paths, all_collision_segments
                else:
                    print("Định dạng đường đi không hợp lệ")
            else:
                print("Không tìm thấy JSON hợp lệ trong phản hồi")
        except Exception as e:
            print(f"Lỗi khi phân tích phản hồi: {e}")
        
        time.sleep(1)
    
    print("Không thể tìm thấy đường đi hợp lệ sau tất cả các lần thử")
    
    # Trả về đường đi gần đây nhất 
    if all_paths:
        return all_paths[-1], all_collision_segments[-1], all_paths, all_collision_segments
    else:
        direct_path = [list(agent_pos), list(goal_pos)]
        has_collision, collision_segments = check_collision(direct_path, obstacles)
        return direct_path, collision_segments, [direct_path], [collision_segments]

def draw_environment(screen, start, goal, obstacles, paths, collision_segments_list):
    screen.fill((255, 255, 255))
    
    # Vẽ vật cản
    for obs in obstacles:
        pygame.draw.polygon(screen, (255, 165, 0), [(x * 100 + 300, -y * 100 + 300) for x, y in obs])
    
    # Vẽ tất cả các đường đi đã thử
    for i in range(len(paths)):
        path = paths[i]
        has_collision = len(collision_segments_list[i]) > 0
        
        if path and len(path) > 1:
            path_color = (255, 150, 150) if has_collision else (150, 150, 255)
            
            # Vẽ các đoạn thẳng kết nối các điểm
            for j in range(len(path) - 1):
                start_point = (path[j][0] * 100 + 300, -path[j][1] * 100 + 300)
                end_point = (path[j+1][0] * 100 + 300, -path[j+1][1] * 100 + 300)
                
                segment_color = path_color
                if has_collision and (j, j+1) in collision_segments_list[i]:
                    segment_color = (255, 0, 0)  
                
                pygame.draw.line(screen, segment_color, start_point, end_point, 3)
            
            # Vẽ các điểm trên đường đi
            for pos in path:
                pygame.draw.circle(screen, (0, 0, 255), (pos[0] * 100 + 300, -pos[1] * 100 + 300), 5)
    
    # Vẽ điểm bắt đầu và kết thúc
    pygame.draw.circle(screen, (0, 255, 0), (start[0] * 100 + 300, -start[1] * 100 + 300), 10)
    pygame.draw.circle(screen, (255, 0, 0), (goal[0] * 100 + 300, -goal[1] * 100 + 300), 10)
    
    # Vẽ chú thích
    font = pygame.font.SysFont('Arial', 16)
    green_text = font.render("Start point", True, (0, 255, 0))
    red_text = font.render("End point", True, (255, 0, 0))
    blue_text = font.render("Valid path", True, (150, 150, 255))
    red_path_text = font.render("Unvalid path", True, (255, 150, 150))
    collision_text = font.render("Collision path", True, (255, 0, 0))
    
    screen.blit(green_text, (20, 20))
    screen.blit(red_text, (20, 45))
    screen.blit(blue_text, (20, 70))
    screen.blit(red_path_text, (20, 95))
    screen.blit(collision_text, (20, 120))
    
    pygame.display.flip()

# Khởi tạo Pygame
pygame.init()
screen = pygame.display.set_mode((600, 600))
pygame.display.set_caption("Path Planning")

# Khai báo môi trường
start = (-1.5, 1.2)
goal = (1.0, -0.5)

obstacles = [
    [(0.0,-0.15), (0.5, 0.3), (0.3, 0.8), (-0.2, 0.4)],
    [(-1.0, 0.0), (-0.6, 0.4), (-1.2, 0.7), (-1.4, 0.3)]
]

print("Lập kế hoạch đường đi...")
final_path, final_collisions, all_paths, all_collision_segments = plan_path(start, goal, obstacles, max_attempts=10)

print(f"Tổng số đường đi đã thử: {len(all_paths)}")

# Vòng lặp chính
running = True
clock = pygame.time.Clock()

while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
    
    draw_environment(screen, start, goal, obstacles, all_paths, all_collision_segments)
    
    clock.tick(30)

pygame.quit()