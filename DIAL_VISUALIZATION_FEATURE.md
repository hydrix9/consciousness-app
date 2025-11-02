# ✨ Interlocking Dial Visualization Feature

## Overview

The **Interlocking Dial Visualization** feature adds a white brush stroke overlay to `--mode generate` that converts your drawing curves into **3D interlocking dial geometry** in real-time.

## What It Does

When you draw curved strokes on the canvas, this feature:
1. **Tracks your stroke points** as you draw
2. **Converts strokes to 3D circular patterns** using the dial system
3. **Renders white overlay curves** showing the dial geometry
4. **Displays dial centers and boundaries** for visual feedback

## Visual Elements

### White Dial Curves
- **Solid white lines** (2px, 200 alpha) showing the 3D curve paths
- Follows the natural flow of your drawing strokes
- Represents the interlocking dial segments

### Dial Boundaries
- **Dashed white circles** (1px, 100 alpha) showing dial extent
- Indicates the radius and center of each generated dial
- Helps visualize the circular nature of the geometry

### Center Points
- **White dots** (4px, 255 alpha) marking dial centers
- Shows the focal point of each dial's rotation
- Useful for understanding dial positioning

## How to Use

### 1. Enable the Feature

```bash
python run.py --mode generate --test-rng --no-eeg --debug
```

### 2. Toggle Visualization

In the Drawing Controls panel:
- ✅ **Check** "Show Interlocking Dials" checkbox
- This enables the white overlay visualization
- Draw curves to see the dial geometry appear!

### 3. Draw and Observe

- **Draw curved strokes** on the canvas
- **Watch white overlays** render the 3D dial geometry
- **See dial boundaries** and center points
- **Multiple strokes** create interlocking dial systems

### 4. Clear and Restart

- Click **"Clear Canvas"** to reset both:
  - Your drawing pixmap
  - The dial visualization overlay
  - All dial geometry data

## Technical Details

### Architecture

```python
class PaintCanvas:
    def __init__(self):
        # Main drawing layer (colored strokes)
        self.pixmap = QPixmap(width, height)
        
        # White dial overlay (transparent background)
        self.dial_overlay = QPixmap(width, height)
        
        # 3D dial geometry system
        self.dial_system = InterlockingDialSystem()
        
        # Current stroke tracking
        self.current_stroke_points = []
        
        # Toggle flag
        self.show_dial_visualization = False
```

### Stroke-to-Dial Conversion

When you release the mouse (stroke end):

```python
def mouseReleaseEvent(self, event):
    # Add final point
    self.current_stroke_points.append((x, y))
    
    # Convert to dial if visualization enabled
    if self.show_dial_visualization and len(points) >= 3:
        dial_id = self.dial_system.add_stroke(
            self.current_stroke_points,
            (canvas_width, canvas_height),
            color_tuple
        )
        
        # Render white overlay
        self._render_dial_visualization()
```

### Rendering Process

The `_render_dial_visualization()` method:

1. **Clears overlay** to transparent
2. **Iterates through dials** in the dial system
3. **Draws curve segments** as connected white lines
4. **Projects 3D to 2D** using orthographic projection
5. **Renders dial boundaries** as dashed circles
6. **Marks center points** with solid white dots

### Paint Event

The canvas composites both layers:

```python
def paintEvent(self, event):
    painter = QPainter(self)
    
    # Draw main colored canvas
    painter.drawPixmap(self.rect(), self.pixmap, self.pixmap.rect())
    
    # Draw white dial overlay (if enabled)
    if self.show_dial_visualization:
        painter.drawPixmap(self.rect(), self.dial_overlay, self.dial_overlay.rect())
```

## Features

### ✅ Real-Time Conversion
- Strokes converted to dials immediately on mouse release
- No lag or processing delay
- Smooth visualization updates

### ✅ Transparent Overlay
- White curves render on top of your colored drawings
- Doesn't interfere with your painting
- Can be toggled on/off anytime

### ✅ 3D Geometry Visualization
- Shows the circular patterns extracted from your curves
- Visualizes the interlocking dial system
- Helps understand the 3D interpretation of 2D strokes

### ✅ Multiple Dial Support
- Each stroke creates a new dial
- Dials can interlock based on proximity
- System tracks all dials independently

## Data Integration

The dial visualization is **purely visual** - it doesn't affect data logging:

- ✅ Your colored drawing actions are logged normally
- ✅ Consciousness layer and pocket dimension tracked
- ✅ RNG and EEG data captured as usual
- ✅ Dial geometry is stored separately for 3D export

The white overlay is **not recorded** in the pixmap - it's a live visualization aid.

## Use Cases

### 1. **Understanding Dial Generation**
See how your curved strokes map to 3D circular patterns

### 2. **Debugging Geometry**
Verify that the dial system is working correctly

### 3. **Creative Feedback**
Get real-time visual feedback on the geometric structure

### 4. **Interlocking Visualization**
See how multiple strokes create interconnected dial systems

## Performance

- **Minimal overhead**: Only renders when strokes complete
- **Transparent overlay**: Fast compositing
- **No data duplication**: Single dial system instance
- **Efficient rendering**: Only draws visible elements

## Limitations

- Requires **at least 3 points** per stroke for dial conversion
- **Orthographic projection** simplifies 3D to 2D (ignores Z-axis)
- **White color only** for clarity and contrast
- **No animation** (yet) - dials are static visualizations

## Future Enhancements

Potential improvements:

1. **Animated rotation** showing dial movement
2. **Color-coded dials** matching stroke colors
3. **Interactive dial manipulation** (drag to rotate)
4. **Export dial geometry** to 3D model formats
5. **RNG-driven animation** based on hardware data

## Testing

Run the test script:

```bash
cd consciousness-app
python test_dial_visualization.py
```

Expected output:
```
✅ Dial system available and ready
✅ PaintCanvas created with dial visualization support
✅ All dial visualization attributes present
✅ All dial visualization methods present
✅ Created dial 1 from test stroke
✅ Dial visualization rendering successful
```

## Code Files Modified

### `src/gui/painting_interface.py`

**Added attributes to `PaintCanvas.__init__`:**
- `dial_overlay`: Transparent pixmap for white curves
- `dial_system`: InterlockingDialSystem instance
- `show_dial_visualization`: Toggle flag
- `current_stroke_points`: Stroke tracking list

**Added methods:**
- `set_dial_visualization(enabled)`: Toggle the feature
- `_render_dial_visualization()`: Render white overlay
- Updated `clear_canvas()`: Clear both layers
- Updated `mousePressEvent()`: Start stroke tracking
- Updated `mouseMoveEvent()`: Add points to stroke
- Updated `mouseReleaseEvent()`: Convert stroke to dial
- Updated `paintEvent()`: Composite both layers

**Added UI controls:**
- `dial_viz_checkbox`: Checkbox to enable/disable
- `toggle_dial_visualization()`: Checkbox handler

**Added imports:**
- `InterlockingDialSystem` from `src.utils.curve_3d`
- `LineToCircleConverter` from `src.utils.curve_3d`
- `DIAL_SYSTEM_AVAILABLE` availability flag

## Summary

The **Interlocking Dial Visualization** feature provides:

✨ **Real-time white overlay** showing 3D dial geometry  
🎨 **Visual feedback** for stroke-to-curve conversion  
🔄 **Toggle-able display** via checkbox  
⚡ **Efficient rendering** with minimal overhead  
🎯 **Clear visualization** of interlocking dial systems  

**Perfect for understanding how your 2D drawings map to 3D consciousness geometry!**
