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

package org.ejml.data;

import java.util.Iterator;

/**
 * Very useful for sparse vector-matrix multiplication
 *
 * Implements ReshapeMatrix so utility functions can be used without any restriction
 *
 * wrapper around (n, 1) {@link DMatrixSparseCSC}
 */
public class DVectorSparse implements Matrix {
    // TODO make this private
    public DMatrixSparseCSC oneDimMatrix;

    public DVectorSparse(int size) {
        oneDimMatrix = new DMatrixSparseCSC(size, 1);
    }

    public DVectorSparse(int size, int arrayLength) {
        oneDimMatrix = new DMatrixSparseCSC(size, 1, arrayLength);
    }

    public DVectorSparse(DVectorSparse original) {
        oneDimMatrix = original.oneDimMatrix.copy();
    }

    public DVectorSparse(double[] values, double zeroElement) {
        this(values.length);

        for (int index = 0; index < values.length; index++) {
            double value = values[index];

            if (value != zeroElement) {
                // safe way ..
                // could also first gather and then do this in one step
                append(index, value);
            }
        }

        oneDimMatrix.indicesSorted = true;
    }

    public DVectorSparse(DMatrixSparseCSC original) {
        assert original.numCols == 1;

        oneDimMatrix = original.copy();
    }

    public int[] nz_indices() {
        return oneDimMatrix.nz_rows;
    }

    public double[] nz_values() {
        return oneDimMatrix.nz_values;
    }

    public int nz_length() {
        return oneDimMatrix.nz_length;
    }

    public boolean isIndicesSorted() {
        return oneDimMatrix.indicesSorted;
    }

    public void setIndicesSorted(boolean sorted) {
        this.oneDimMatrix.indicesSorted = sorted;
    }

    public int size() {
        return oneDimMatrix.numRows;
    }

    @Override
    public DVectorSparse copy() {
        return new DVectorSparse(this);
    }

    public void set(DVectorSparse original) {
        this.oneDimMatrix.set(original.oneDimMatrix);
    }
    
    @Override
    public int getNumRows() { return oneDimMatrix.getNumRows();
    }

    @Override
    public int getNumCols() {
        return oneDimMatrix.getNumCols();
    }

    @Override
    public void zero() {
        oneDimMatrix.zero();
    }

    @Override
    public DVectorSparse create(int numRows, int numCols) {
        assert numRows == 1 || numCols == 1;
        int size = numRows == 1 ? numCols : numRows;
        return new DVectorSparse(size);
    }

    @Override
    public void set(Matrix original) {
        assert original instanceof DVectorSparse;
        oneDimMatrix.set(original);
    }

    @Override
    public void print() {
        StringBuilder resultBuilder = new StringBuilder("[");
        for (int i = 0; i < size(); i++) {
            if (i != 0) {
                resultBuilder.append(",");
            }
            if (isAssigned(i)) {
                resultBuilder.append(get(i));
            } else {
                resultBuilder.append("*");
            }
        }
        System.out.println(resultBuilder.append("]").toString());
    }

    @Override
    public void print(String format) {
        oneDimMatrix.print(format);
    }

    @Override
    public MatrixType getType() {
        throw new UnsupportedOperationException("There is no matrix type for sparse vectors yet");
    }

    public void append(int index, double value) {
        int entry = nz_length();
        if (nz_length() == nz_indices().length) {
            growMaxLength(nz_length()*2+1, true);
        }

        nz_indices()[entry] = index;
        nz_values()[entry] = value;
        oneDimMatrix.col_idx[1]++;
        oneDimMatrix.nz_length++;
    }

    public void printNonZero() {
        this.oneDimMatrix.printNonZero();
    }

    public boolean isAssigned(int index) {
        return oneDimMatrix.isAssigned(index, 0);
    }

    public double get(int index) {
        return oneDimMatrix.get(index, 0, Double.NaN);
    }

    public double get(int index, double fallBackValue) {
        return oneDimMatrix.get(index, 0, fallBackValue);
    }

    public double unsafe_get(int index,double fallBackValue) {
        return oneDimMatrix.unsafe_get(index, 0, fallBackValue);
    }

    public void set(int index, double val) {
        this.oneDimMatrix.set(index, 0, val);
    }

    public void unsafe_set(int index, double val) {
        this.oneDimMatrix.unsafe_set(index, 0, val);
    }

    public void remove(int index) {
        this.oneDimMatrix.remove(index, 0);
    }

    public void reshape(int size) {
        this.oneDimMatrix.reshape(size, 1);
    }

    public void reshape(int size, int array_length) {
        this.oneDimMatrix.reshape(size, 1, array_length);
    }

    public void growMaxLength( int arrayLength, boolean preserveValues) {
        this.oneDimMatrix.growMaxLength(arrayLength, preserveValues);
    }

    public void sortIndices() {
        this.oneDimMatrix.sortIndices(null);
    }

    public void copyStructure( DVectorSparse orig) {
        this.oneDimMatrix.copyStructure(orig.oneDimMatrix);
    }

    public Iterator<RealValue> createIterator() {
        return new Iterator<>() {
            RealValue coordinate = new RealValue();
            int nz_index = 0; // the index of the non-zero value and row

            @Override
            public boolean hasNext() {
                return nz_index < oneDimMatrix.nz_length;
            }

            @Override
            public RealValue next() {
                coordinate.index = oneDimMatrix.nz_rows[nz_index];
                coordinate.value = oneDimMatrix.nz_values[nz_index];
                nz_index++;
                return coordinate;
            }
        };

    }

    @Override
    public DVectorSparse createLike() {
        return new DVectorSparse(this.size());
    }

    public void setMatrix(DMatrixSparseCSC matrix) {
        assert matrix.numCols == 1;

        this.oneDimMatrix = matrix;
    }

    /**
     * Value of an element in a sparse vector
     */
    class RealValue {
        /** The coordinate */
        public int index;
        /** The value of the coordinate */
        public double value;
    }
}
