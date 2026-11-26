# Pseudocode
```c
PROGRAM nearest_neighbor_upscale

    // ---------- Part 1: setup ----------
    LOAD image "input" INTO original
    SET oldWidth  = width of original
    SET oldHeight = height of original

    SET newWidth  = the width you want
    SET newHeight = the height you want

    CREATE empty grid result WITH newHeight rows AND newWidth columns

    // ---------- Part 2: lookup lists ----------
    CREATE list sourceCol WITH newWidth slots
    FOR col FROM 0 TO newWidth - 1
        SET answer = (col * oldWidth) / newWidth      // whole-number division, decimals dropped
        IF answer > oldWidth - 1 THEN
            SET answer = oldWidth - 1                 // never go past the last column
        END IF
        SET sourceCol[col] = answer
    END FOR

    CREATE list sourceRow WITH newHeight slots
    FOR row FROM 0 TO newHeight - 1
        SET answer = (row * oldHeight) / newHeight
        IF answer > oldHeight - 1 THEN
            SET answer = oldHeight - 1
        END IF
        SET sourceRow[row] = answer
    END FOR

    // ---------- Part 3: fill the new image ----------
    FOR row FROM 0 TO newHeight - 1                   // <-- OpenMP goes on this loop
        SET fromRow = sourceRow[row]
        FOR col FROM 0 TO newWidth - 1
            SET fromCol = sourceCol[col]
            SET result[row][col] = original[fromRow][fromCol]
        END FOR
    END FOR

    // ---------- Part 4: finish ----------
    SAVE result AS "output"
    FREE original, result, sourceCol, sourceRow

END PROGRAM
```
What each part does

Part 1: setup. You load the picture, write down its size, decide the new size, and make an empty grid to fill. Nothing is calculated yet.

Part 2: lookup lists. These are the “which original column/row do I copy from?” answers, computed once and stored. sourceCol[3] means: “new column 3 copies from this original column.” The IF is a safety net so you never point past the edge of the original.

Part 3: filling. There are two loops, one inside the other:

The outer loop walks down the rows of the new image.
The inner loop walks across the columns of that row.
For each spot, you look up where it comes from and copy that pixel.

This is the “reading a page line by line” idea from before.

Part 4: finish. Save the result and give back the memory you borrowed.

Tracing it on a tiny example

Original 3×3 enlarged to 5×5, so oldWidth = oldHeight = 3 and newWidth = newHeight = 5.

original:
    10  20  30
    40  50  60
    70  80  90

Part 2 builds the lists:

col	(col × 3) / 5	sourceCol[col]
0	0 / 5	0
1	3 / 5	0
2	6 / 5	1
3	9 / 5	1
4	12 / 5	2

sourceRow comes out the same: [0, 0, 1, 1, 2].

Part 3, the first few trips through the loops:

row = 0 → fromRow = sourceRow[0] = 0
    col = 0 → fromCol = 0 → result[0][0] = original[0][0] = 10
    col = 1 → fromCol = 0 → result[0][1] = original[0][0] = 10
    col = 2 → fromCol = 1 → result[0][2] = original[0][1] = 20
    col = 3 → fromCol = 1 → result[0][3] = original[0][1] = 20
    col = 4 → fromCol = 2 → result[0][4] = original[0][2] = 30
row = 1 → fromRow = sourceRow[1] = 0
    ... same as row 0 → 10 10 20 20 30
row = 2 → fromRow = sourceRow[2] = 1
    col = 0 → result[2][0] = original[1][0] = 40
    ... → 40 40 50 50 60

Final result:

    10  10  20  20  30
    10  10  20  20  30
    40  40  50  50  60
    40  40  50  50  60
    70  70  80  80  90

Following the pseudocode by hand like this, with a pencil, is the best way to make sure you understand it before writing C.
. 