/*
 * Copyright (c) 2009-2020, Peter Abeles. All Rights Reserved.
 *
 * This file is part of Efficient Java Matrix Library (EJML).
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package org.ejml.masks;

import org.ejml.data.*;

/**
 * Helper class to create the corresponding mask based on a matrix or primitive array
 */
public class DMasks {
    // TODO: use builder?

    public static Mask of(double[] values, boolean negated) {
        return of(values, negated, 0);
    }

    public static Mask of(double[] values, boolean negated, double zeroElement) {
        return new PrimitiveDMask(values, negated, zeroElement);
    }

    public static Mask of(DMatrixD1 matrix, boolean negated) {
        return of(matrix, negated, 0);
    }

    public static Mask of(DMatrixD1 matrix, boolean negated, double zeroElement) {
        return new PrimitiveDMask(matrix.data, matrix.numCols, negated, zeroElement);
    }

    public static Mask of(DMatrixSparseCSC matrix, boolean negated, boolean structural, double zeroElement){
        if (structural) {
            return new SparseStructuralMask(matrix, negated);
        }
        else {
            return new SparseDMask(matrix, negated, zeroElement);
        }
    }

    // structural masks cannot have a zeroElement
    public static Mask of(DMatrixSparseCSC matrix, boolean negated, boolean structural) {
        return of(matrix, negated, structural, 0);
    }
}