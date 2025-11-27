def extract_features(events, metadata, viewport):
    hover_time_why_different = 0
    hover_time_product_desc = 0
    mouse_moves = 0
    min_timestamp = events[0]['timestamp']
    max_timestamp = events[-1]['timestamp']
    
    for event in events:
        if event['type'] == 'hover':
            elem_text = event['data']['element']['textContent']
            duration = event['data'].get('duration', 0)
            if "Why It's Different" in elem_text:
                hover_time_why_different += duration
            if "product description" in elem_text.lower() or "bean" in elem_text.lower():
                hover_time_product_desc += duration
                
        if event['type'] == 'mouse_move':
            mouse_moves += 1
    
    session_time = (max_timestamp - min_timestamp) / 1000  # milliseconds to seconds
    
    features = {
        'hover_time_why_different': hover_time_why_different / 1000,  # seconds
        'hover_time_product_desc': hover_time_product_desc / 1000,
        'mouse_move_count': mouse_moves,
        'session_duration': metadata.get('sessionDuration', 0) / 1000,
        'scroll_y': viewport.get('scrollY', 0),
        'viewport_width': viewport.get('width', 0),
        'viewport_height': viewport.get('height', 0),
        'device_pixel_ratio': viewport.get('devicePixelRatio', 1),
        # Add more features as needed
    }
    
    return features
