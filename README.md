# Knots
it's not not knots


# Image extraction
It would be tedious to convert example knots to our domain model
by hand. `knots/serde/raster` contains tools to convert idiomatic
knot sketches  (see [The Knot Book](https://www.math.cuhk.edu.hk/course_builder/1920/math4900e/Adams--The%20Knot%20Book.pdf) for examples) into our
domain models. It works as an image pipeline that vectorizes the sketches
and infers knot connectivity:
