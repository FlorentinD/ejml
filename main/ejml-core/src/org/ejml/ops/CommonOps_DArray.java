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

package org.ejml.ops;

import org.ejml.data.DMatrixSparseCSC;
import org.ejml.masks.PrimitiveDMask;

/**
 * @author Florentin Doerre
 *
 * methods for manipulating primitive double arrays
 */
public class CommonOps_DArray {
    // TODO: support for mask and accumulator (when needed)

    public static double[] apply( double[] v, DUnaryOperator func ) {
        return apply(v, v, func);
    }

    /**
     * Apply a unary function on every element in v and store it in out
     */
    public static double[] apply(double[] v, double[] out, DUnaryOperator func) {
        if (v.length != out.length) {
            throw new IllegalArgumentException("Input and Output vectors must have the same length");
        }

        for (int i = 0; i < v.length; i++) {
            out[i] = func.apply(v[i]);
        }

        return v;
    }

    /**
     * Assigns the elements in the sparse matrix to the primitive vector.
     * ! Assuming the Matrix has only 1 row .
     *
     * @param v Vector of size N
     * @param w Vector stored as a 1-dim matrix
     * @return
     */
    public static double[] assign( double[] v, DMatrixSparseCSC w) {
        if (w.numRows != 1 && w.numCols != 1) {
            throw new IllegalArgumentException("Matrix is not a vector");
        }

        if (w.numRows == 1) {
            // column vector
            for (int i = 0; i < w.numCols; i++) {
                if (w.col_idx[i] != w.col_idx[i+1]) {
                    v[i] = w.nz_values[w.col_idx[i]];
                }
            }
        } else {
            // row vector
            for (int i = 0; i < w.nz_length; i++) {
                v[w.nz_rows[i]] = w.nz_values[i];
            }
        }

        return v;
    }

    public static double[] elementWiseMult(double[] a, double[] b, double[] output, DBinaryOperator mult) {
        assert(a.length == b.length && b.length == output.length);

        for (int i = 0; i < a.length; i++) {
            output[i] = mult.apply(a[i], b[i]);
        }

        return output;
    }

    public static double[] elementWiseAdd(double[] a, double[] b, double[] output, DMonoid add) {
        // dense version -> no difference between elementWiseMult or elementWiseAdd
        return elementWiseMult(a, b, output, add.func);
    }

    public static double reduceScalar(double[] v, DMonoid monoid) {
        double result = monoid.id;
        for (double value : v) {
            result = monoid.func.apply(result, value);
        }

        return result;
    }

    /**
     *
     * @param v     (Input) vector
     * @param monoid Operator to use for reduction and intial value
     * @param mask !! differs from normal mask (here for which elements should be included from v)
     * @return      accumulated Value
     */
    public static double reduceScalar(double[] v, DMonoid monoid, PrimitiveDMask mask) {
        double result = monoid.id;
        for (int i = 0; i < v.length; i++) {
            if (mask.isSet(i)) {
                result = monoid.func.apply(result, v[i]);
            }
        }

        return result;
    }
}
