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
        model="llama3-8b-8192",  
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

def generate_prompt(agent_pos, goal_pos, obstacles, attempt=1, collision_info=None, obstacle_info=None):
    collision_text = ""
    if collision_info and len(collision_info) > 0:
        collision_details = []
        for i, j in collision_info:
            collision_details.append(f"Collision: Between points {i} and {j} in previous path")
        
        if collision_details:
            collision_text = "Collision information:\n" + "\n".join(collision_details) + "\n\n"
    
    obstacles_text = ""
    if obstacle_info:
        obstacles_text = f"""
        Important obstacle movement information:
        - Obstacles are oscillating back and forth horizontally
        - First obstacle moves with amplitude {obstacle_info[0]['amplitude']} and period {obstacle_info[0]['period']} seconds
        - Second obstacle moves with amplitude {obstacle_info[1]['amplitude']} and period {obstacle_info[1]['period']} seconds
        """
    
    prompt = f"""
    You are a path planning assistant. Given start coordinates, goal coordinates, and oscillating obstacles, 
    you need to return a valid path as a JSON object.
    
    Start: {agent_pos}
    Goal: {goal_pos}
    Obstacles (quadrilaterals): {json.dumps(obstacles)}
    {obstacles_text}
    
    IMPORTANT: I need ONLY a JSON object in the format {{"Trajectory": [[x1, y1], [x2, y2], ...]}}
    DO NOT include any code, explanation, or other text. ONLY return the JSON.
    
    This is attempt #{attempt}.
    {collision_text}
    DO NOT repeat the collision in the previous path.

    The path should:
    1. Start at exactly {agent_pos}
    2. End at exactly {goal_pos}
    3. Avoid all obstacles considering they are OSCILLATING horizontally (moving back and forth)
    4. Use waypoints as needed to navigate around obstacles safely
    5. Consider timing the movement to pass when obstacles are at favorable positions
    
    Example of correct response:
    {{"Trajectory": [[-1.5, 1.2], [-0.8, 0.8], [-0.35, 0.5], [0.0, -0.25], [1.0, -0.5]]}}
    
    You have to provide your own answer that avoids the oscillating obstacles.
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

def calculate_oscillating_position(original_position, amplitude, period, phase, elapsed_time):
    """Tính toán vị trí của đối tượng dao động theo thời gian."""
    offset = amplitude * math.sin(2 * math.pi * elapsed_time / period + phase)
    return [original_position[0] + offset, original_position[1]]

def move_oscillating_polygon(polygon, amplitude, period, phase, elapsed_time):
    offset = amplitude * math.sin(2 * math.pi * elapsed_time / period + phase)
    return [[p[0] + offset, p[1]] for p in polygon]

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

def check_collision_with_oscillating_obstacles(path, obstacles, obstacle_params, agent_speed):
    collision_segments = []
    has_collision = False
    
    # Ước tính thời gian cho mỗi đoạn đường
    segment_times = []
    total_distance = 0
    for i in range(len(path) - 1):
        segment_distance = math.sqrt((path[i+1][0] - path[i][0])**2 + (path[i+1][1] - path[i][1])**2)
        total_distance += segment_distance
        segment_times.append(segment_distance / agent_speed)
    
    # Kiểm tra va chạm cho mỗi đoạn đường
    current_time = 0
    for i in range(len(path) - 1):
        segment_has_collision = False
        
        # Chia đoạn đường thành nhiều điểm nhỏ để kiểm tra chi tiết hơn
        num_checks = 10
        for j in range(num_checks + 1):
            check_time = current_time + (j / num_checks) * segment_times[i]
            
            # Vị trí agent tại thời điểm kiểm tra
            t_ratio = j / num_checks
            check_point = [
                path[i][0] + t_ratio * (path[i+1][0] - path[i][0]),
                path[i][1] + t_ratio * (path[i+1][1] - path[i][1])
            ]
            
            # Kiểm tra va chạm với từng vật cản được di chuyển đến vị trí tương ứng
            for obs_idx, obs in enumerate(obstacles):
                param = obstacle_params[obs_idx]
                moved_obs = move_oscillating_polygon(
                    obs, param['amplitude'], param['period'], param['phase'], check_time
                )
                
                if is_point_inside_polygon(check_point, moved_obs):
                    print(f"Phát hiện va chạm tại điểm {check_point} ở thời điểm {check_time:.2f}s")
                    segment_has_collision = True
                    has_collision = True
                    break
            
            if segment_has_collision:
                break
        
        if segment_has_collision:
            collision_segments.append((i, i+1))
        
        current_time += segment_times[i]
    
    return has_collision, collision_segments

def plan_path(agent_pos, goal_pos, obstacles, obstacle_params, agent_speed, max_attempts=5):
    all_paths = []  
    all_collision_segments = []  
    
    for attempt in range(max_attempts):
        print(f"Lần thử {attempt+1}/{max_attempts}")
        collision_info = all_collision_segments[-1] if all_collision_segments else None
        prompt = generate_prompt(agent_pos, goal_pos, obstacles, attempt+1, collision_info, obstacle_params)
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
                        
                    has_collision, collision_segments = check_collision_with_oscillating_obstacles(
                        path, obstacles, obstacle_params, agent_speed
                    )
                    
                    # Lưu đường đi và các đoạn va chạm
                    all_paths.append(path)
                    all_collision_segments.append(collision_segments)
                    
                    if not has_collision:
                        print("Tìm thấy đường đi hợp lệ:", path)
                        return path, collision_segments, all_paths, all_collision_segments
                else:
                    print("Định dạng đường đi không hợp lệ")
                    all_paths.append([])
                    all_collision_segments.append([])
            else:
                print("Không tìm thấy JSON hợp lệ trong phản hồi")
                all_paths.append([])
                all_collision_segments.append([])
        except Exception as e:
            print(f"Lỗi khi phân tích phản hồi: {e}")
            all_paths.append([])
            all_collision_segments.append([])
        
        time.sleep(1)
    
    print("Không thể tìm thấy đường đi hợp lệ sau tất cả các lần thử")
    
    valid_paths = [p for p in all_paths if p]
    if valid_paths:
        idx = all_paths.index(valid_paths[-1])
        return valid_paths[-1], all_collision_segments[idx], all_paths, all_collision_segments
    else:
        direct_path = [list(agent_pos), list(goal_pos)]
        has_collision, collision_segments = check_collision_with_oscillating_obstacles(
            direct_path, obstacles, obstacle_params, agent_speed
        )
        return direct_path, collision_segments, [direct_path], [collision_segments]

def draw_environment(screen, start_pos, goal, obstacles, paths, collision_segments_list, current_obstacles, agent_pos, elapsed_time):
    screen.fill((255, 255, 255))
    
    # Vẽ vật cản hiện tại
    for obs in current_obstacles:
        pygame.draw.polygon(screen, (255, 165, 0), [(x * 100 + 300, -y * 100 + 300) for x, y in obs])
    
    # Vẽ tất cả các đường đi đã thử
    for i in range(len(paths)):
        path = paths[i]
        if not path:  
            continue
            
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
    
    # Vẽ điểm bắt đầu, kết thúc và điểm hiện tại của agent
    pygame.draw.circle(screen, (0, 255, 0), (start_pos[0] * 100 + 300, -start_pos[1] * 100 + 300), 7)  
    pygame.draw.circle(screen, (255, 0, 0), (goal[0] * 100 + 300, -goal[1] * 100 + 300), 10)
    pygame.draw.circle(screen, (0, 0, 255), (agent_pos[0] * 100 + 300, -agent_pos[1] * 100 + 300), 10) 
    
    # Vẽ chú thích
    font = pygame.font.SysFont('Arial', 16)
    green_text = font.render("Start point", True, (0, 255, 0))
    red_text = font.render("End point", True, (255, 0, 0))
    blue_text = font.render("Agent point", True, (0, 0, 255))
    valid_path_text = font.render("Valid path", True, (150, 150, 255))
    invalid_path_text = font.render("Unvalid path", True, (255, 150, 150))
    collision_text = font.render("Collision path", True, (255, 0, 0))
    time_text = font.render(f"Time: {elapsed_time:.2f}s", True, (0, 0, 0))
    
    screen.blit(green_text, (20, 20))
    screen.blit(red_text, (20, 45))
    screen.blit(blue_text, (20, 70))
    screen.blit(valid_path_text, (20, 95))
    screen.blit(invalid_path_text, (20, 120))
    screen.blit(collision_text, (20, 145))
    screen.blit(time_text, (20, 170))
    
    pygame.display.flip()

def calculate_agent_position(path, elapsed_time, agent_speed):
    """Tính toán vị trí hiện tại của agent dựa trên thời gian đã trôi qua và tốc độ."""
    if not path or len(path) < 2:
        return path[0] if path else [0, 0]
    
    # Tính toán tổng chiều dài đường đi
    total_distance = 0
    segment_distances = []
    for i in range(len(path) - 1):
        distance = math.sqrt((path[i+1][0] - path[i][0])**2 + (path[i+1][1] - path[i][1])**2)
        segment_distances.append(distance)
        total_distance += distance
    
    distance_traveled = agent_speed * elapsed_time
    
    # Xác định agent đang ở đoạn nào của đường đi
    if distance_traveled >= total_distance:
        return path[-1]  # Đã đến đích
    
    current_distance = 0
    for i in range(len(segment_distances)):
        next_distance = current_distance + segment_distances[i]
        if distance_traveled <= next_distance:
            segment_progress = (distance_traveled - current_distance) / segment_distances[i]
            return [
                path[i][0] + segment_progress * (path[i+1][0] - path[i][0]),
                path[i][1] + segment_progress * (path[i+1][1] - path[i][1])
            ]
        current_distance = next_distance
    
    return path[-1]  

# Khởi tạo Pygame
pygame.init()
screen = pygame.display.set_mode((600, 600))
pygame.display.set_caption("Path Planning với Vật Cản Dao Động")

# Khai báo môi trường
start = (-1.5, 1.2)
goal = (1.0, -0.5)

obstacles = [
    [(0.0,-0.15), (0.5, 0.3), (0.3, 0.8), (-0.2, 0.4)],
    [(-1.0, 0.0), (-0.6, 0.4), (-1.2, 0.7), (-1.4, 0.3)]
]

obstacle_params = [
    {'amplitude': 0.5, 'period': 4.0, 'phase': 0},     
    {'amplitude': 0.4, 'period': 5.0, 'phase': math.pi}  
]
 
agent_speed = 0.8

print("Lập kế hoạch đường đi...")
final_path, final_collisions, all_paths, all_collision_segments = plan_path(
    start, goal, obstacles, obstacle_params, agent_speed, max_attempts=5
)

print(f"Tổng số đường đi đã thử: {len(all_paths)}")

# Vòng lặp chính
running = True
clock = pygame.time.Clock()
start_time = time.time()
reached_goal = False

while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
    
    elapsed_time = time.time() - start_time
    
    current_obstacles = []
    for i, obs in enumerate(obstacles):
        param = obstacle_params[i]
        moved_obs = move_oscillating_polygon(
            obs, param['amplitude'], param['period'], param['phase'], elapsed_time
        )
        current_obstacles.append(moved_obs)
    
    # Tính vị trí hiện tại của agent
    current_agent_pos = calculate_agent_position(final_path, elapsed_time, agent_speed)
    
    # Vẽ môi trường với vật cản dao động và agent di chuyển
    draw_environment(screen, start, goal, obstacles, all_paths, all_collision_segments, 
                    current_obstacles, current_agent_pos, elapsed_time)
    
    # Kiểm tra xem agent đã đến đích chưa
    distance_to_goal = math.sqrt((current_agent_pos[0] - goal[0])**2 + (current_agent_pos[1] - goal[1])**2)
    if distance_to_goal < 0.1 and not reached_goal:
        print(f"Agent đã đến đích sau {elapsed_time:.2f} giây!")
        reached_goal = True

    clock.tick(30) 

pygame.quit()