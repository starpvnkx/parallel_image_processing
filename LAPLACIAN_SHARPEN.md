# Sequential Pseudocode.
```c
INCLUDE standard libraries (stdio, stdlib, stdint)
INCLUDE stb_image, etc.

DEFINE kernel weights
center = 5
top = bot = left = right = -1

int main(void) {
    READ image file headers to get width and height
    ALLOCATE memory for input_image array of size (width * height) as uint8_t
    uint8_t input_image = malloc(sizeof() * input_image * width * height)
    ALLOCATE memory for output_image arraay of size (width * height) as uint8_t
    uint8_t output_image = malloc(sizeof() * output_image * width * height)

    for y from 1 up to (height - 2) {
        for x from 1 up to (width - 2){
            CALCULATE 1D indices for the center and neighbors
            center_index = (y*width) + x
            top_index = ((y - 1) * width) + x
            bot_index = ((y + 1) * width) + x
            right_index = (y * width) + (x + 1)
            left_index = (y * width) + (x - 1)

            READ the pixel values
            center_val = input_image[center_index]
            top_val = input_image[top_index]
            .
            .
            .

            APPLY kernel math
            new_val = (center_val * 5) - top_val - bottom_val - left_val - right_val

            CLAMP the new value
            new_val = min(max(new_val, 0), 255)

            STORE the new value
            output_image[center_index] = new_val 
        }
    }

    WRITE output_image array to a new image file
    FREE images
    free(input_image)
    free(output_image)
}
```