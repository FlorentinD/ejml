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

import org.ejml.data.DGrowArray;
import org.ejml.data.DMatrixRMaj;
import org.ejml.data.DMatrixSparseCSC;
import org.ejml.data.IGrowArray;
import org.ejml.masks.Mask;
import org.ejml.ops.DBinaryOperator;

import javax.annotation.Nullable;

import static org.ejml.UtilEjml.adjust;
import static org.ejml.UtilEjml.checkSameShape;

public class MaskUtil_DSCC {
    // for applying mask and accumulator (output gets overwritten)
    // ! assumes that the mask is already applied to the output .. e.g. unset fields not even computed (and not assigned)
    static DMatrixSparseCSC combineOutputs(DMatrixSparseCSC output, DBinaryOperator accum, DMatrixSparseCSC initialOutput) {
        if (initialOutput != null) {
            // memory overhead .. maybe also can reuse something?
            DMatrixSparseCSC combinedOutput = output.createLike();
            // instead of "semiRing.add" this could be a dedicated accumulator
            add(initialOutput, output, combinedOutput, accum, null, null);
            // is the previous result of C gc-able? (should be)
            output = combinedOutput;
        }
        return output;
    }

    static DMatrixRMaj combineOutputs(DMatrixRMaj output, DMatrixRMaj initialOutput, Mask mask, @Nullable DBinaryOperator accum) {
        if (initialOutput != null) {
            checkSameShape(initialOutput, output, true);

            // TODO: operate on a bitset/boolean[] here -> also just one for-loop needed
            for (int col = 0; col < output.getNumCols(); col++) {
                for (int row = 0; row < output.numRows; row++) {
                    if (mask.isSet(row, col)) {
                        if (accum != null) {
                            // combine previous value and computed value
                            output.unsafe_set(row, col, accum.apply(output.get(row, col), initialOutput.get(row, col)));
                        }
                        // else output value just keeps as it is
                    } else {
                        // just use previous value as it shouldnt be computed in the first place
                        output.unsafe_set(row, col, initialOutput.get(row, col));
                    }
                }
            }
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
    public static void add(DMatrixSparseCSC A, DMatrixSparseCSC B, DMatrixSparseCSC C, DBinaryOperator accum,
                           @Nullable IGrowArray gw, @Nullable DGrowArray gx) {
        double[] x = adjust(gx, A.numRows);
        int[] w = adjust(gw, A.numRows, A.numRows);

        C.indicesSorted = false;
        C.nz_length = 0;

        for (int col = 0; col < A.numCols; col++) {
            C.col_idx[col] = C.nz_length;

            // always take the values of A
            // second as x[row] would be the first argument
            multAddColA(A, col, C, col + 1, (a, b) -> b, x, w);
            // accum values of A and B
            multAddColA(B, col, C, col + 1, accum, x, w);

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
                                   DBinaryOperator accum,
                                   double x[], int w[]) {
        int idxA0 = A.col_idx[colA];
        int idxA1 = A.col_idx[colA + 1];

        for (int j = idxA0; j < idxA1; j++) {
            int row = A.nz_rows[j];

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
