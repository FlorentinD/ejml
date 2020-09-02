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

/**
 * @author Florentin Doerre
 *
 * methods for manipulating primitive double arrays
 */
public class CommonOps_DArray {
    // TODO: support for mask and accumulator (when needed)

    public static double[] apply(double[] v, DUnaryOperator func) {
        for (int i = 0; i < v.length; i++) {
            v[i] = func.apply(v[i]);
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

    public static double reduceScalar(double[] v, DMonoid monoid) {
        double result = monoid.id;
        for (int i = 0; i < v.length; i++) {
            result = monoid.func.apply(result, v[i]);
        }

        return result;
    }
}
