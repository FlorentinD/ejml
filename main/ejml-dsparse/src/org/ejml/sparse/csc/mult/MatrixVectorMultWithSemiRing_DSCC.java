/*
 * Copyright (c) 2009-2018, Peter Abeles. All Rights Reserved.
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

package org.ejml.sparse.csc.mult;

import org.ejml.data.DMatrixSparseCSC;
import org.ejml.masks.PrimitiveDMask;
import org.ejml.ops.DBinaryOperator;
import org.ejml.ops.DSemiRing;
import org.ejml.sparse.csc.MaskUtil_DSCC;

import javax.annotation.Nullable;
import java.util.Arrays;

/**
 * based on MartrixVectorMult_DSCC
 */
public class MatrixVectorMultWithSemiRing_DSCC {
    /**
     * output = A*b
     *
     * @param A (Input) Matrix
     * @param b (Input) vector
     * @param output (Output) vector
     * @param mask Mask for specifying which entries should be overwritten
     */
    public static double[] mult(DMatrixSparseCSC A, double b[], double output[], DSemiRing semiRing,
                            @Nullable PrimitiveDMask mask, @Nullable DBinaryOperator accumulator) {
        double[] initialOutput = MaskUtil_DSCC.useInitialOutput(mask, accumulator, output) ? output.clone() : null;
        if (mask != null) {
            mask.compatible(output);
        }

        Arrays.fill(output, semiRing.add.id);

        return multAdd(A, b, output, initialOutput, semiRing, mask, accumulator);
    }

    public static double[] mult(DMatrixSparseCSC A, double b[], double output[], DSemiRing semiRing) {
        return mult(A, b, output, semiRing, null, null);
    }

    /**
     * output = initialOutput + A*b
     * ! difference to normal `multAdd` without semiRing: apply + on result of A*b
     *
     * @param A           (Input) Matrix
     * @param b           (Input) vector
     * @param output      (Output) vector
     * @param mask        Mask for specifying which entries should be overwritten
     * @param semiRing
     * @param accumulator (Optional) accumulator for output + (A*b), elso use `add` from the semiRing
     */
    public static double[] multAdd(DMatrixSparseCSC A, double[] b, double[] output, @Nullable double[] initialOutput,
                               DSemiRing semiRing, @Nullable PrimitiveDMask mask, @Nullable DBinaryOperator accumulator) {
        for (int k = 0; k < A.numCols; k++) {
            int idx0 = A.col_idx[k];
            int idx1 = A.col_idx[k + 1];

            for (int indexA = idx0; indexA < idx1; indexA++) {
                if (mask == null || mask.isSet(A.nz_rows[indexA])) {
                    output[A.nz_rows[indexA]] = semiRing.add.func.apply(
                            output[A.nz_rows[indexA]],
                            semiRing.mult.func.apply(A.nz_values[indexA], b[k]));
                }
            }
        }

        // initialOutput + output
        return MaskUtil_DSCC.combineOutputs(initialOutput, output, mask, accumulator);
    }

    /**
     * output = a<sup>T</sup>*B
     *
     * @param a       (Input) vector
     * @param B       (Input) Matrix
     * @param output       (Output) vector
     * @param mask Mask for specifying which entries should be overwritten
     */
    public static double[] mult(double a[], DMatrixSparseCSC B, double output[], DSemiRing semiRing,
                            @Nullable PrimitiveDMask mask, @Nullable DBinaryOperator accumulator) {
        double[] initialOutput = MaskUtil_DSCC.useInitialOutput(mask, accumulator, output) ? output.clone() : null;
        if (mask != null) {
            mask.compatible(output);
        }

        for (int k = 0; k < B.numCols; k++) {
            if (mask == null || mask.isSet(k)) {
                int idx0 = B.col_idx[k];
                int idx1 = B.col_idx[k + 1];


                double sum = semiRing.add.id;
                for (int indexB = idx0; indexB < idx1; indexB++) {
                    sum = semiRing.add.func.apply(sum, semiRing.mult.func.apply(a[B.nz_rows[indexB]], B.nz_values[indexB]));
                }
                output[k] = sum;
            }
        }

        return MaskUtil_DSCC.combineOutputs(initialOutput, output, mask, accumulator);
    }

    public static double[] mult(double a[], DMatrixSparseCSC B, double c[], DSemiRing semiRing) {
        return mult(a, B, c, semiRing, null, null);
    }

    /**
     * scalar = A<sup>T</sup>*B*C
     *
     * @param a       (Input) vector
     * @param offsetA Input) first index in vector a
     * @param B       (Input) Matrix
     * @param c       (Output) vector
     * @param offsetC (Output) first index in vector c
     */
    public static double innerProduct(double a[], int offsetA,
                                      DMatrixSparseCSC B,
                                      double c[], int offsetC, DSemiRing semiRing) {
        if (a.length - offsetA < B.numRows)
            throw new IllegalArgumentException("Length of 'a' isn't long enough");
        if (c.length - offsetC < B.numCols)
            throw new IllegalArgumentException("Length of 'c' isn't long enough");

        double output = 0;

        for (int k = 0; k < B.numCols; k++) {
            int idx0 = B.col_idx[k];
            int idx1 = B.col_idx[k + 1];

            double sum = 0;
            for (int indexB = idx0; indexB < idx1; indexB++) {
                sum = semiRing.add.func.apply(sum, semiRing.mult.func.apply(a[offsetA + B.nz_rows[indexB]], B.nz_values[indexB]));
            }
            output = semiRing.add.func.apply(output, semiRing.mult.func.apply(sum, c[offsetC + k]));
        }

        return output;
    }
}
