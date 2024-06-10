from heapq import heappush, heappop

import time, os


def clear_screen():
    os.system('cls' if os.name == 'nt' else 'clear')

def manhattan_distance(cell1, cell2):
    return abs(cell1[0] - cell2[0]) + abs(cell1[1] - cell2[1])

def astar(grid, start1, goal1, start2, goal2):
    rows, cols = len(grid), len(grid[0])
    open_list = []
    closed_list = set()
    parent = {}
    g_score = {(start1, start2): 0}
    f_score = {(start1, start2): manhattan_distance(start1, goal1) + manhattan_distance(start2, goal2)}

    heappush(open_list, (f_score[(start1, start2)], (start1, start2)))

    while open_list:
        current = heappop(open_list)[1]
        if current[0] == goal1 and current[1] == goal2:
            path = []
            while current in parent:
                path.append(current)
                current = parent[current]
            path.append((start1, start2))
            path.reverse()
            return path

        closed_list.add(current)
        for dx1, dy1, dx2, dy2 in [(0, 1, 0, 1), (0, -1, 0, -1), (1, 0, 1, 0), (-1, 0, -1, 0),
                                   (0, 1, 0, 0), (0, -1, 0, 0), (1, 0, 0, 0), (-1, 0, 0, 0),
                                   (0, 0, 0, 1), (0, 0, 0, -1), (0, 0, 1, 0), (0, 0, -1, 0)]:
            neighbor1 = (current[0][0] + dx1, current[0][1] + dy1)
            neighbor2 = (current[1][0] + dx2, current[1][1] + dy2)
            if (0 <= neighbor1[0] < rows and 0 <= neighbor1[1] < cols and
                0 <= neighbor2[0] < rows and 0 <= neighbor2[1] < cols and
                grid[neighbor1[0]][neighbor1[1]] != 1 and grid[neighbor2[0]][neighbor2[1]] != 1 and
                (neighbor1 != neighbor2 or neighbor1 == goal1) and (neighbor1, neighbor2) not in closed_list):
                g = g_score[current] + 1
                f = g + manhattan_distance(neighbor1, goal1) + manhattan_distance(neighbor2, goal2)
                if (neighbor1, neighbor2) not in [i[1] for i in open_list] or g < g_score.get((neighbor1, neighbor2), float('inf')):
                    parent[(neighbor1, neighbor2)] = current
                    g_score[(neighbor1, neighbor2)] = g
                    f_score[(neighbor1, neighbor2)] = f
                    heappush(open_list, (f, (neighbor1, neighbor2)))

    return None
def print_grid(grid, pos1=None, pos2=None, path1=None, path2=None):
    rows, cols = len(grid), len(grid[0])
    for i in range(rows):
        for j in range(cols):
            if pos1 == (i, j):
                print("1 ", end="")
            elif pos2 == (i, j):
                print("2 ", end="")
            elif path1 and (i, j) in path1:
                print(". ", end="")
            elif path2 and (i, j) in path2:
                print(". ", end="")
            elif grid[i][j] == 1:
                print("█ ", end="")
            else:
                print("· ", end="")
        print()

# Sample grid with an obstacle
grid = [
    [0,0,0,0,0,0,  1,1,1,1,1,1,1, 0, 1,1,1,1,1,1,1, 0, 1,1,1,1,1,1,1, 0,0,0,0,0,0],
    [0,0,0,0,0,0,  1,1,1,1,1,1,1, 0, 1,1,1,1,1,1,1, 0, 1,1,1,1,1,1,1, 0,0,0,0,0,0],
    [0,0,0,0,0,0,  1,1,1,1,1,1,1, 0, 1,1,1,1,1,1,1, 0, 1,1,1,1,1,1,1, 0,0,0,0,0,0],
    [0,0,0,0,0,0,  1,1,1,1,1,1,1, 0, 1,1,1,1,1,1,1, 0, 1,1,1,1,1,1,1, 0,0,0,0,0,0],
    [0,0,0,0,0,0,  1,1,1,1,1,1,1, 0, 1,1,1,1,1,1,1, 0, 1,1,1,1,1,1,1, 0,0,0,0,0,0],
    [0,0,0,0,0,0,  1,1,1,1,1,1,1, 0, 1,1,1,1,1,1,1, 0, 1,1,1,1,1,1,1, 0,0,0,0,0,0],
    [0,0,0,0,0,0,  1,1,1,1,1,1,1, 0, 1,1,1,1,1,1,1, 0, 1,1,1,1,1,1,1, 0,0,0,0,0,0],
    [0,0,0,0,0,0,  1,1,1,1,1,1,1, 0, 1,1,1,1,1,1,1, 0, 1,1,1,1,1,1,1, 0,0,0,0,0,0],
    [0,0,0,0,0,0,  1,1,1,1,1,1,1, 0, 1,1,1,1,1,1,1, 0, 1,1,1,1,1,1,1, 0,0,0,0,0,0],
    [0,0,0,0,0,0,  1,1,1,1,1,1,1, 0, 1,1,1,1,1,1,1, 0, 1,1,1,1,1,1,1, 0,0,0,0,0,0],
    [0,0,0,0,0,0,  1,1,1,1,1,1,1, 0, 1,1,1,1,1,1,1, 0, 1,1,1,1,1,1,1, 0,0,0,0,0,0],
    [0,0,0,0,0,0,  1,1,1,1,1,1,1, 0, 1,1,1,1,1,1,1, 0, 1,1,1,1,1,1,1, 0,0,0,0,0,0],
    [0,0,0,0,0,0,  1,1,1,1,1,1,1, 0, 1,1,1,1,1,1,1, 0, 1,1,1,1,1,1,1, 0,0,0,0,0,0],
    [0,0,0,0,0,0,  1,1,1,1,1,1,1, 0, 1,1,1,1,1,1,1, 0, 1,1,1,1,1,1,1, 0,0,0,0,0,0],
    [0,0,0,0,0,0,  1,1,1,1,1,1,1, 0, 1,1,1,1,1,1,1, 0, 1,1,1,1,1,1,1, 0,0,0,0,0,0],
    [0,0,0,0,0,0,  1,1,1,1,1,1,1, 0, 1,1,1,1,1,1,1, 0, 1,1,1,1,1,1,1, 0,0,0,0,0,0],

    [0,0,0,0,0,0,  0,0,0,0,0,0,0, 0, 0,0,0,0,0,0,0, 0, 0,0,0,0,0,0,0, 0,0,0,0,0,0],

    [0,0,0,0,0,0,  1,1,1,1,1,1,1, 0, 1,1,1,1,1,1,1, 0, 1,1,1,1,1,1,1, 0,0,0,0,0,0],
    [0,0,0,0,0,0,  1,1,1,1,1,1,1, 0, 1,1,1,1,1,1,1, 0, 1,1,1,1,1,1,1, 0,0,0,0,0,0],
    [0,0,0,0,0,0,  1,1,1,1,1,1,1, 0, 1,1,1,1,1,1,1, 0, 1,1,1,1,1,1,1, 0,0,0,0,0,0],
    [0,0,0,0,0,0,  1,1,1,1,1,1,1, 0, 1,1,1,1,1,1,1, 0, 1,1,1,1,1,1,1, 0,0,0,0,0,0],
    [0,0,0,0,0,0,  1,1,1,1,1,1,1, 0, 1,1,1,1,1,1,1, 0, 1,1,1,1,1,1,1, 0,0,0,0,0,0],
    [0,0,0,0,0,0,  1,1,1,1,1,1,1, 0, 1,1,1,1,1,1,1, 0, 1,1,1,1,1,1,1, 0,0,0,0,0,0],
    [0,0,0,0,0,0,  1,1,1,1,1,1,1, 0, 1,1,1,1,1,1,1, 0, 1,1,1,1,1,1,1, 0,0,0,0,0,0],
    [0,0,0,0,0,0,  1,1,1,1,1,1,1, 0, 1,1,1,1,1,1,1, 0, 1,1,1,1,1,1,1, 0,0,0,0,0,0],
    [0,0,0,0,0,0,  1,1,1,1,1,1,1, 0, 1,1,1,1,1,1,1, 0, 1,1,1,1,1,1,1, 0,0,0,0,0,0],
    [0,0,0,0,0,0,  1,1,1,1,1,1,1, 0, 1,1,1,1,1,1,1, 0, 1,1,1,1,1,1,1, 0,0,0,0,0,0],
    [0,0,0,0,0,0,  1,1,1,1,1,1,1, 0, 1,1,1,1,1,1,1, 0, 1,1,1,1,1,1,1, 0,0,0,0,0,0],
    [0,0,0,0,0,0,  1,1,1,1,1,1,1, 0, 1,1,1,1,1,1,1, 0, 1,1,1,1,1,1,1, 0,0,0,0,0,0],
    [0,0,0,0,0,0,  1,1,1,1,1,1,1, 0, 1,1,1,1,1,1,1, 0, 1,1,1,1,1,1,1, 0,0,0,0,0,0],
    [0,0,0,0,0,0,  1,1,1,1,1,1,1, 0, 1,1,1,1,1,1,1, 0, 1,1,1,1,1,1,1, 0,0,0,0,0,0],
    [0,0,0,0,0,0,  1,1,1,1,1,1,1, 0, 1,1,1,1,1,1,1, 0, 1,1,1,1,1,1,1, 0,0,0,0,0,0],
    [0,0,0,0,0,0,  1,1,1,1,1,1,1, 0, 1,1,1,1,1,1,1, 0, 1,1,1,1,1,1,1, 0,0,0,0,0,0],
    [0,0,0,0,0,0,  1,1,1,1,1,1,1, 0, 1,1,1,1,1,1,1, 0, 1,1,1,1,1,1,1, 0,0,0,0,0,0],
    [0,0,0,0,0,0,  1,1,1,1,1,1,1, 0, 1,1,1,1,1,1,1, 0, 1,1,1,1,1,1,1, 0,0,0,0,0,0],

    [0,0,0,0,0,0,  0,0,0,0,0,0,0, 0, 0,0,0,0,0,0,0, 0, 0,0,0,0,0,0,0, 0,0,0,0,0,0],
    [0,0,0,0,0,0,  0,0,0,0,0,0,0, 0, 0,0,0,0,0,0,0, 0, 0,0,0,0,0,0,0, 0,0,0,0,0,0],
]

start1 = (20, 2)
goal1 = (7, 22)
start2 = (13, 34)
goal2 = (5, 12)


print("Grid with obstacle:")
# mark both the goals as available
grid[goal1[0]][goal1[1]] = grid[goal2[0]][goal2[1]] = 0
print_grid(grid, start1, start2)

path = astar(grid, start1, goal1, start2, goal2)
if path:
    path1 = [cell[0] for cell in path]
    path2 = [cell[1] for cell in path]
    
    print("\nAnimation:")
    for i in range(len(path)):
        clear_screen()
        print(f"Step {i+1}:")
        print_grid(grid, path1[i], path2[i], path1[:i+1], path2[:i+1])
        time.sleep(0.1) 

    print("\nNo paths found.")