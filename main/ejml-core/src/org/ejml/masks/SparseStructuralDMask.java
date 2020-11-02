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

/**
 * only looking if the entry is assigned in the source(disregarding the actual stored value)
 * ! it does not copy the input matrix -> changing the matrix structure will also affect the mask
 *
 */
public class SparseStructuralDMask extends Mask {
    // TODO make independent of data-type
    private final DMatrixSparseCSC matrix;

    private int indexedColumn = -1;
    // int[] instead of boolean[] to avoid clearing on multiple setActiveColumns()
    // if row in indexed column -> rowIndicesInIndexedColumn[row] == col
    private int[] rowIndicesInIndexedColumn;

    public SparseStructuralDMask( DMatrixSparseCSC matrix, boolean negated, boolean indexFirstColumn ) {
        super(negated);
        this.matrix = matrix;
        this.rowIndicesInIndexedColumn = new int[matrix.numRows];
        if (indexFirstColumn) {
            setIndexColumn(0);
        }
    }

    @Override
    public boolean isSet(int row, int col) {
        if (col != indexedColumn) {
            return negated ^ matrix.isAssigned(row, col);
        } else {
            return negated ^ (rowIndicesInIndexedColumn[row] - 1 == col);
        }
    }

    @Override
    public int getNumCols() {
        return matrix.getNumCols();
    }

    @Override
    public int getNumRows() {
        return matrix.getNumRows();
    }

    @Override
    public void setIndexColumn( int col ) {
        if (indexedColumn != col) {
            this.indexedColumn = col;
            for (int i = matrix.col_idx[col]; i < matrix.col_idx[col+1]; i++) {
                rowIndicesInIndexedColumn[matrix.nz_rows[i]] = col + 1;
            }
        }
    }

    public static class Builder extends MaskBuilder<SparseStructuralDMask> {
        private DMatrixSparseCSC matrix;
        private boolean indexFirstColumn;

        public Builder(DMatrixSparseCSC matrix) {
            this.matrix = matrix;
        }

        public Builder withIndexFirstColumn( boolean indexFirstColumn) {
            this.indexFirstColumn = indexFirstColumn;
            return this;
        }

        @Override
        public SparseStructuralDMask build() {
            return new SparseStructuralDMask(matrix, negated, indexFirstColumn);
        }
    }
}
