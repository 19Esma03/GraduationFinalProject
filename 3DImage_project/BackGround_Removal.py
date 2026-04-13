from rembg import remove

def remove_background(image):
    print("Background removing...")
    
    try:
        image_nobg = remove(image)
        return image_nobg
    except Exception as e:
        print(f"Background Delete Error: {e}")
        return image