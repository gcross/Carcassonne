{-# LANGUAGE UnicodeSyntax #-}

import Control.Monad
import Control.Parallel.Strategies
import Data.Graph.Inductive
import Data.Word

import ContractionSearch

graph :: Word → ContractionGraph
graph nb =
    mkGraph
        ((0,0):[(i,2) | i ← [1..2]])
        ((1,2,nb*2):[(0,i,2) | i ← [1..2]])

main = do
    let results = parMap rseq (findMinimalContractionCost . graph) [1,2,3]
    forM_ [0,1,2] $ \i →
        putStrLn $
            ("B = " ++ show (i+1) ++ ": ")
            ++
            show (results !! i)
