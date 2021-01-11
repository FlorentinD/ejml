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

public class PrimitiveDMask extends Mask {
    // TODO: Check if creating dense boolean mask is worth it (currently converting to boolean on the fly)
    private final double[] values;
    // alternatively declare extra separate masks for primitive and dense DMatrix
    private final int numCols;
    //
    private final double zeroElement;

    public PrimitiveDMask( double[] values, int numCols, boolean negated, double zeroElement ) {
        // for dense structures they cannot be used for structural masks
        super(negated);
        this.values = values;
        this.numCols = numCols;
        this.zeroElement = zeroElement;
    }

    @Override
    public boolean isSet( int row, int col ) {
        // XOR as negated flips the mask flag
        return negated ^ (values[row*numCols + col] != zeroElement);
    }

    public double getZeroElement() { return zeroElement;}

    @Override
    public int getNumCols() {
        return numCols;
    }

    @Override
    public int getNumRows() {
        return values.length/numCols;
    }

    @Override
    public void setIndexColumn( int column ) {}

    @Override
    public boolean isSet( int index ) {
        return negated ^ (values[index] != zeroElement);
    }

    @Override
    public void compatible( int size ) {
        if (size != values.length) {
            throw new MatrixDimensionException(String.format(
                    "Mask of length %d cannot be applied to vector of length %d",
                    values.length, size
            ));
        }
    }

    public static class Builder extends MaskBuilder<PrimitiveDMask> {
        private double[] values;
        private int numCols = 1;
        private double zeroElement = 0;

        public Builder( double[] values ) {
            this.values = values;
        }

        public Builder withNumCols( int numCols ) {
            this.numCols = numCols;
            return this;
        }

        public Builder withZeroElement( double zeroElement ) {
            this.zeroElement = zeroElement;
            return this;
        }

        @Override
        public PrimitiveDMask build() {
            return new PrimitiveDMask(values, numCols, negated, zeroElement);
        }
    }
}

