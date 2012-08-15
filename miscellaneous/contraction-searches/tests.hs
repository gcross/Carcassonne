-- Language extensions {{{
{-# LANGUAGE UnicodeSyntax #-}
-- }}}

-- Imports {{{
import Data.Graph.Inductive
import Data.List
import Data.Word
import Debug.Trace (trace)

import Test.Framework
import Test.Framework.Providers.HUnit
import Test.Framework.Providers.QuickCheck2
import Test.HUnit
import Test.QuickCheck

import ContractionSearch
-- }}}

-- Helper Functions {{{
dont_echo :: α → α -- {{{
dont_echo x = x
-- }}}
dont_echoWithLabel :: String → α → α -- {{{
dont_echoWithLabel label x = x
-- }}}
echo :: Show α ⇒ α → α -- {{{
echo x = trace (show x) x
-- }}}
echoWithLabel :: Show α ⇒ String → α → α -- {{{
echoWithLabel label x = trace (label ++ " " ++ show x) x
-- }}}
-- }}}

tests = -- {{{
    [testGroup "findMinimalContractionCost" -- {{{
        [testCase "null" $ -- {{{
            findMinimalContractionCost empty @?= 0
         -- }}}
        ,testProperty "one node" $ \d → -- {{{
            findMinimalContractionCost
                (mkGraph [(0,d)] [])
            == 0
         -- }}}
        ,testProperty "two nodes" $ \d1 d2 b → -- {{{
            findMinimalContractionCost
                (mkGraph [(1,d1),(2,d2)] [(1,2,b)])
            == d1+d2+b
         -- }}}
        ,testProperty "three nodes" $ \d1 d2 d3 b12 b23 b31 -- {{{
          → findMinimalContractionCost
                (mkGraph
                    [(1,d1),(2,d2),(3,d3)]
                    [(1,2,b12),(2,3,b23),(3,1,b31)]
                )
            == minimum
                [maximum [b12+b23+b31+d1+d2,b23+b31+d1+d2+d3]
                ,maximum [b23+b31+b12+d2+d3,b31+b12+d2+d3+d1]
                ,maximum [b31+b12+b23+d3+d1,b12+b23+d3+d1+d2]
                ]
         -- }}}
        ,testProperty "four nodes, random labels" $ \ -- {{{
          d1 d2 d3 d4
          b12 b13 b14 b23 b24 b34
          →
           let b :: Word → Word → Word
               b 1 2 = b12
               b 2 1 = b12
               b 1 3 = b13
               b 3 1 = b13
               b 1 4 = b14
               b 4 1 = b14
               b 2 3 = b23
               b 3 2 = b23
               b 2 4 = b24
               b 4 2 = b24
               b 3 4 = b34
               b 4 3 = b34
               d :: Word → Word
               d 1 = d1
               d 2 = d2
               d 3 = d3
               d 4 = d4
               costForBandwidths :: [Word] → Word
               costForBandwidths permutation = minimum $
                [maximum
                    [b_12+b_13+b_14+b_23+b_24+d_1+d_2
                    ,b_13+b_23+b_14+b_24+b_34+d_1+d_2+d_3
                    ,b_14+b_24+b_34+d_1+d_2+d_3+d_4
                    ]
                ,maximum
                    [b_12+b_13+b_14+b_23+b_24+d_1+d_2
                    ,b_13+b_23+b_14+b_24+b_34+d_3+d_4
                    ,b_13+b_14+b_23+b_24+d_1+d_2+d_3+d_4
                    ]
                ,maximum
                    [b_12+b_13+b_14+b_23+b_24+d_1+d_2
                    ,b_13+b_23+b_14+b_24+b_34+d_1+d_2+d_4
                    ,b_13+b_23+b_34+d_1+d_2+d_3+d_4
                    ]
                ]
                where
                 r :: Word → Word
                 r i = genericIndex permutation (i-1)
                 b_12 = b (r 1) (r 2)
                 b_13 = b (r 1) (r 3)
                 b_14 = b (r 1) (r 4)
                 b_23 = b (r 2) (r 3)
                 b_24 = b (r 2) (r 4)
                 b_34 = b (r 3) (r 4)
                 d_1 = d (r 1)
                 d_2 = d (r 2)
                 d_3 = d (r 3)
                 d_4 = d (r 4)
               graph =
                mkGraph
                [(1,d1),(2,d2),(3,d3),(4,d4)]
                [(1,2,b12)
                ,(1,3,b13)
                ,(1,4,b14)
                ,(2,3,b23)
                ,(2,4,b24)
                ,(3,4,b34)
                ] :: Gr Word Word
           in
            findMinimalContractionCost graph
            == minimum [costForBandwidths permutation | permutation ← permutations [1..4]]
         -- }}}
        ]
    ] -- }}}
-- }}}

main = defaultMain tests