{-# LANGUAGE UnicodeSyntax #-}

import Control.Monad
import Control.Parallel.Strategies
import Data.Graph.Inductive
import Data.Word

import ContractionSearch

graph :: Word → ContractionGraph
graph nb =
    mkGraph
        ([(i,0) | i ← [1,3..9]] ++ [(i,1) | i ← [2,4..8]])
        ((8,1,nb):[(i,i+1,nb) | i ← [1..7]] ++ [(i,9,1) | i ← [2,4..8]])

main = do
    let results = parMap rseq (findMinimalContractionCost . graph) [1,2,3]
    forM_ [0,1,2] $ \i →
        putStrLn $
            ("B = " ++ show (i+1) ++ ": ")
            ++
            show (results !! i)
