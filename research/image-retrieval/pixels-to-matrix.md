# From pixels to a feature matrix

**Master’s thesis progress · November 17, 2026 · Ruoshui Li**  
Mathematics, Saint Louis University · Advisor: Vahan Huroyan

How does one location on a bird become one column of a feature matrix?

## The forward pass

I followed a northern cardinal photograph from CUB-200-2011 through VGG16 with ImageNet weights. For a 448 × 448 input, the output of the last convolutional block before the final pooling operation has 512 channels on a 28 × 28 grid. Flattening the grid in row order gives a matrix **A** with 512 rows and 784 columns. Each row covers one feature across the image. Each column collects all 512 feature responses at one location.

The grid cell near the beak is at row 9, column 13. Counting rows and columns from one, its index in the flattened grid is (9 − 1) × 28 + 13 = **237**. Here are three of the 512 responses in that column, rounded to two decimals.

| Feature map | Response |
| --- | ---: |
| 458 | 26.67 |
| 15 | 2.97 |
| 178 | 0.00 |

[Machine-readable values](results/cardinal-column.csv)

## A worked pooling example

In bilinear pooling, entry (i, j) of **AAᵀ** sums the products of the responses of features i and j over all positions. A strong sum can reflect a pattern that appears repeatedly.

Consider two hypothetical birds. For each, a grass feature responds with (2, 2, 2, 2, 0). A beak feature responds with (0, 0, 0, 0, 1) for the first bird, and (0, 0, 0, 0, 2) for the second. The diagonal entries of **AAᵀ** are (16, 1) and (16, 4). The largest entry, 16, stays the same. The smaller entry carries the distinction in this example.

The numbers in this example were chosen to isolate the effect. The table above comes from the actual VGG16 forward pass. My next question is how changing the weights on singular directions affects which features count in a comparison.

[Research overview](README.md)  ·  [Background reading](REFERENCES.md)
