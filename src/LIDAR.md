# LiDAR Data Format Documentation

## Overview

The LiDAR system generates training data for machine learning by recording agent sensor readings along with actions taken. Each data point represents a complete snapshot of the agent's sensory and motor state at a specific moment.

## JSON Data Structure

```json
{
  "timestamp": 45.67,
  "agent_position": [100.5, 200.3],
  "agent_rotation": 1.57,
  "agent_velocity": [5.0, -2.0],
  "distances": [25.0, 30.0, 15.0, 18.2, 45.7, ...],
  "angles": [0.0, 1.57, 3.14, 4.71, 6.28, ...],
  "hit_points": [
    [125.5, 200.3],
    [130.5, 230.3],
    [85.5, 200.3],
    ...
  ],
  "action_taken": "move_up"
}
```

## Field Reference

| Field | Type | Units | Description |
|-------|------|-------|-------------|
| `timestamp` | number | seconds | Game time when recorded |
| `agent_position` | [number, number] | world units | Agent's [x, y] position |
| `agent_rotation` | number | radians | Agent's facing direction |
| `agent_velocity` | [number, number] | units/sec | Agent's [x, y] velocity |
| `distances` | number[] | world units | Ray distances to obstacles |
| `angles` | number[] | radians | Absolute ray angles |
| `hit_points` | [number, number][] | world units | Ray collision points |
| `action_taken` | string \| null | - | Action performed |

## Coordinate System

- **Origin**: Bottom-left corner (standard Cartesian)
- **X-axis**: Positive = right
- **Y-axis**: Positive = up
- **Rotation**: 0 = facing right, π/2 = facing up (counter-clockwise)

## Ray Configuration

Default LiDAR setup:
- **Range**: 100 units maximum detection distance
- **Rays**: 72 rays (5° increments)
- **Coverage**: 360° full circle
- **Frequency**: 10 Hz updates
- **Sensor Radius**: 4 units (collision avoidance offset)

## Actions

Standard keyboard mappings:
- `"move_up"`: Arrow Up pressed
- `"move_down"`: Arrow Down pressed  
- `"move_left"`: Arrow Left pressed
- `"move_right"`: Arrow Right pressed
- `"no_action"`: No movement keys pressed

## Data Quality

### Filtering
- Records only when at least one ray distance > 6.1 units
- Limits recordings to 10,000 per session
- Uses FIFO replacement when limit exceeded

### Statistics
Export includes:
```
Exported 1500 LiDAR recordings to lidar_data_1640995200.json
Action distribution: {"move_up": 400, "move_right": 350, "no_action": 750}  
Sample distances: [45.2, 67.8, 23.1, 89.4, 12.7]
```

## Usage Examples

### Python Data Loading
```python
import json
import numpy as np

# Load recorded data
with open('lidar_data_1640995200.json', 'r') as f:
    recordings = json.load(f)

# Extract features and labels
features = []
labels = []

for record in recordings:
    # Combine sensor and state data
    feature = (
        record['distances'] +           # Ray distances
        record['agent_position'] +      # [x, y] position  
        [record['agent_rotation']] +    # Rotation
        record['agent_velocity']        # [x, y] velocity
    )
    features.append(feature)
    labels.append(record['action_taken'])

X = np.array(features)
y = np.array(labels)
```

### Neural Network Input
```python
# Typical input shape for 72-ray LiDAR:
# - distances: 72 values
# - position: 2 values (x, y)
# - rotation: 1 value
# - velocity: 2 values (x, y)
# Total: 77 input features

input_size = 77
hidden_size = 128
output_size = 5  # 4 movement actions + no_action

model = nn.Sequential(
    nn.Linear(input_size, hidden_size),
    nn.ReLU(),
    nn.Linear(hidden_size, output_size),
    nn.Softmax(dim=1)
)
```

## File Naming

Exported files use timestamp naming:
```
lidar_data_{unix_timestamp}.json
```

Example: `lidar_data_1640995200.json` (exported on 2021-12-31)