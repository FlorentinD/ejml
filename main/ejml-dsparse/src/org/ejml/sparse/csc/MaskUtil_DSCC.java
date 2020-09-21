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

package org.ejml.sparse.csc;

import org.ejml.data.*;
import org.ejml.masks.Mask;
import org.ejml.masks.PrimitiveDMask;
import org.ejml.ops.DBinaryOperator;
import org.jetbrains.annotations.Nullable;

import static org.ejml.UtilEjml.adjust;
import static org.ejml.UtilEjml.checkSameShape;

public class MaskUtil_DSCC {
    private static final DBinaryOperator SECOND = (x, y) -> y;

    // for applying mask and accumulator (output gets overwritten)
    // ! assumes that the mask is already applied to the output .. e.g. unset fields not even computed (and not assigned)
    // ! the mask is only needed for `apply` there the mask is not applied to the result structure before
    public static DMatrixSparseCSC combineOutputs(DMatrixSparseCSC output, @Nullable DMatrixSparseCSC initialOutput, @Nullable Mask mask, @Nullable DBinaryOperator accum) {
        if (initialOutput != null) {
            if(accum == null) {
                // e.g. just take the newly computed value
                accum = SECOND;
            }

            // memory overhead .. maybe also can reuse something?
            DMatrixSparseCSC combinedOutput = output.createLike();
            // instead of "semiRing.add" this could be a dedicated accumulator
            add(initialOutput, output, combinedOutput, mask, accum, null, null);
            // is the previous result of C gc-able? (should be)
            output = combinedOutput;
        }
        return output;
    }

    static DMatrixSparseCSC combineOutputs(DMatrixSparseCSC output, @Nullable DMatrixSparseCSC initialOutput, @Nullable DBinaryOperator accum) {
        return combineOutputs(output, initialOutput, null, accum);
    }

    static DMatrixRMaj combineOutputs(DMatrixRMaj output, @Nullable DMatrixRMaj initialOutput, @Nullable PrimitiveDMask mask, @Nullable DBinaryOperator accum, boolean maskApplied) {
        if (initialOutput != null) {
            checkSameShape(initialOutput, output, true);

            if (accum == null) {
                // e.g. just take the newly computed value
                accum = SECOND;
            }

            // TODO simplify to mask.isSet(index)
            for (int col = 0; col < output.getNumCols(); col++) {
                for (int row = 0; row < output.numRows; row++) {
                    if (mask == null || mask.isSet(row, col)) {
                        initialOutput.unsafe_set(row, col, accum.apply(initialOutput.get(row, col), output.get(row, col)));
                    }
                }
            }

            output = initialOutput;
        } else if (mask != null && !maskApplied) {
            // in case the mask wasnt applied during computation f.i. reduceRowWise
            for (int i = 0; i < output.data.length; i++) {
                // zero unwanted elements
                if (!mask.isSet(i)) {
                    output.data[i] = mask.getZeroElement();
                }
            }
        }

        return output;
    }

    public static double[] combineOutputs(@Nullable double[] initialOutput, double[] output, @Nullable PrimitiveDMask mask, @Nullable  DBinaryOperator accum) {
        // TODO also use maskApplied here
        if (initialOutput != null) {
            if(accum == null) {
                // e.g. just take the newly computed value
                accum = SECOND;
            }

            for (int i = 0; i < output.length; i++) {
                if (mask == null || mask.isSet(i)) {
                    initialOutput[i] = accum.apply(initialOutput[i], output[i]);
                }
            }

            output = initialOutput;
        }

        return output;
    }

    /**
     * Performs matrix addition:<br>
     * C = A + B
     *
     * .. f.i. to combine intialOutput and computedOutput (needed for masks and accumulators in GraphHBLAS)
     * @param A  Matrix
     * @param B  Matrix
     * @param C  Output matrix.
     * @param accum accumulator
     * @param gw (Optional) Storage for internal workspace.  Can be null.
     * @param gx (Optional) Storage for internal workspace.  Can be null.
     */
    public static void add(DMatrixSparseCSC A, DMatrixSparseCSC B, DMatrixSparseCSC C, @Nullable Mask mask, DBinaryOperator accum,
                           @Nullable IGrowArray gw, @Nullable DGrowArray gx) {
        double[] x = adjust(gx, A.numRows);
        int[] w = adjust(gw, A.numRows, A.numRows);

        C.indicesSorted = false;
        C.nz_length = 0;

        for (int col = 0; col < A.numCols; col++) {
            C.col_idx[col] = C.nz_length;

            // always take the values of A
            // second as x[row] would be the first argument
            multAddColA(A, col, C, col + 1, null, (a, b) -> b, x, w);
            // accum values of A and B (if entry is set in mask)
            multAddColA(B, col, C, col + 1, mask, accum, x, w);

            // take the values in the dense vector 'x' and put them into 'C'
            int idxC0 = C.col_idx[col];
            int idxC1 = C.col_idx[col + 1];

            for (int i = idxC0; i < idxC1; i++) {
                C.nz_values[i] = x[C.nz_rows[i]];
            }
        }
        C.col_idx[A.numCols] = C.nz_length;
    }

    /**
     * Performs the performing operation x = x + A(:,i)
     * for applying a accumulator
     */
    public static void multAddColA(DMatrixSparseCSC A, int colA,
                                   DMatrixSparseCSC C, int mark,
                                   @Nullable Mask mask,
                                   DBinaryOperator accum,
                                   double[] x, int[] w) {
        int idxA0 = A.col_idx[colA];
        int idxA1 = A.col_idx[colA + 1];

        for (int j = idxA0; j < idxA1; j++) {
            int row = A.nz_rows[j];
            // TODO: are there cases, where the mask is checked twice this way?
            if (mask == null || mask.isSet(row, mark - 1)) {
                if (w[row] < mark) {
                    if (C.nz_length >= C.nz_rows.length) {
                        C.growMaxLength(C.nz_length * 2 + 1, true);
                    }

                    w[row] = mark;
                    C.nz_rows[C.nz_length] = row;
                    C.col_idx[mark] = ++C.nz_length;
                    x[row] = A.nz_values[j];
                } else if (accum != null) {
                    // if it is null .. x[row] can just stay the same
                    x[row] = accum.apply(x[row], A.nz_values[j]);
                }
            }
        }
    }

    // TODO: remove replace flag and just decide based on accumulator and intialOutput? (replace flag seems redundant)
    //       drawback: user has to specify accumulator everytime (e.g. not default to SECOND)

    /**
     * Check if initialOutput matrix needs to be cached for later merge with actual result
     *
     * ! mask.replace takes precedence before existing accumlator
     */
    public static <T extends Matrix> @Nullable T maybeCacheInitialOutput(@Nullable Mask mask, @Nullable DBinaryOperator accumulator, @Nullable T initialOutput) {
        if (initialOutput != null && ((mask != null && !mask.replace) || (accumulator != null && mask == null))) {
            return initialOutput.copy();
        }
        else {
            return null;
        }
    }

    // primitive version
    public static @Nullable double[] maybeCacheInitialOutput(@Nullable Mask mask, @Nullable DBinaryOperator accumulator, @Nullable double[] initialOutput) {
        if (initialOutput != null && ((mask != null && !mask.replace) || (accumulator != null && mask == null))) {
            return initialOutput.clone();
        }
        else {
            return null;
        }
    }
}
