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


import org.ejml.MatrixDimensionException;
import org.ejml.data.Matrix;

/**
 * Mask used for specifying which matrix entries should be computed
 */
public abstract class Mask {
    // useful for sparse matrices, as actual negation would be costly and result in dense masks
    public final boolean negated;

    protected Mask( boolean negated ) {
        this.negated = negated;
    }

    public abstract boolean isSet(int row, int col);

    public abstract int getNumCols();

    public abstract int getNumRows();

    public void print() {
        StringBuilder result = new StringBuilder();
        for (int row = 0; row < getNumRows(); row++) {
            for (int col = 0; col < getNumCols(); col++) {
                result.append(isSet(row, col) ? "+ " : "- ");
            }
            result.append(System.lineSeparator());
        }

        System.out.println(result);
    }

    /**
     * Checks whether the dimensions of the mask and matrix match
     * @param matrix the mask is applied to
     */
    public void compatible(Matrix matrix) {
        if (matrix.getNumCols() != getNumCols() || matrix.getNumRows() != getNumRows()) {
            throw new MatrixDimensionException(String.format(
                    "Mask of (%d, %d) cannot be applied for matrix (%d, %d)",
                    getNumRows(), getNumCols(), matrix.getNumCols(), matrix.getNumCols()
            ));
        }
    }

    // TODO: use an Iterator as it should be faster as stepping can be used -> no need to call for each entry?
    //      Problem .. dense matrices are row-based, whereas existing sparse matrix format is column based
    //public abstract Iterator<BMatrixSparseCSC.CoordinateBoolValue>
}
