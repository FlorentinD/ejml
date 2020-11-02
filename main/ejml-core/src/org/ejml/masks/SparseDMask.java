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

import org.ejml.data.DMatrixSparseCSC;

import java.util.Arrays;

public class SparseDMask extends Mask {
    protected final DMatrixSparseCSC matrix;
    protected final double zeroElement;

    private int indexedColumn = -1;
    // int[] instead of boolean[] to avoid clearing on multiple setActiveColumns()
    // if row in indexed column -> rowIndicesInIndexedColumn[row] == nz_index + 1
    private int[] rowIndicesInIndexedColumn;

    public SparseDMask( DMatrixSparseCSC matrix, boolean negated, double zeroElement, boolean indexFirstColumn ) {
        super(negated);
        this.matrix = matrix;
        this.zeroElement = zeroElement;
        // TODO allow reusing the array
        this.rowIndicesInIndexedColumn = new int[matrix.numRows];

        if (indexFirstColumn) {
            setIndexColumn(0);
        }
    }

    @Override
    public boolean isSet(int row, int col) {
        if (col != indexedColumn) {
            return negated ^ (matrix.unsafe_get(row, col) != zeroElement);
        } else {
            int nz_index = rowIndicesInIndexedColumn[row] - 1;
            if (nz_index < 0) {
                // no entry in the matrix
                return negated;
            } else {
                return negated ^ (matrix.nz_values[nz_index] != zeroElement);
            }

        }
    }

    @Override
    public int getNumCols() {
        return matrix.numCols;
    }

    @Override
    public int getNumRows() {
        return matrix.numRows;
    }

    @Override
    public void setIndexColumn( int col ) {
        if (indexedColumn != col) {
            // clear column
            Arrays.fill(rowIndicesInIndexedColumn, 0);
            this.indexedColumn = col;
            for (int i = matrix.col_idx[col]; i < matrix.col_idx[col+1]; i++) {
                rowIndicesInIndexedColumn[matrix.nz_rows[i]] = i + 1;
            }
        }
    }

    public static class Builder extends MaskBuilder<SparseDMask> {
        private DMatrixSparseCSC matrix;
        private double zeroElement = 0;
        private boolean indexFirstColumn = false;

        public Builder(DMatrixSparseCSC matrix) {
            this.matrix = matrix;
        }

        public Builder withZeroElement(double zeroElement) {
            this.zeroElement = zeroElement;
            return this;
        }

        public Builder withIndexFirstColumn(boolean indexFirstColumn) {
            this.indexFirstColumn = indexFirstColumn;
            return this;
        }

        @Override
        public SparseDMask build() {
            return new SparseDMask(matrix, negated, zeroElement, indexFirstColumn);
        }
    }
}
