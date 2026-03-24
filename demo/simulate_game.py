from vision import image_to_tictactoe_grid
from tictactoe_engine import find_best_move

def print_grid(grid):
    for row in grid:
        print("| " + " | ".join(c if c != " " else " " for c in row) + " |")

if __name__ == "__main__":
    img_path = input("Path image (ex: tests/images/test.png): ").strip()

    grid = image_to_tictactoe_grid(img_path)
    print("\nDetected grid:")
    print_grid(grid)

    best_move, player_letter, win, is_grid_complete = find_best_move(grid)

    print("\nEngine output:")
    print("best_move:", best_move)
    print("player_letter:", player_letter)
    print("win:", win)
    print("is_grid_complete:", is_grid_complete)
