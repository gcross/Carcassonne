-- Language extensions {{{
{-# LANGUAGE UnicodeSyntax #-}
-- }}}

module ContractionSearch where

-- Imports {{{
import Control.Monad

import Data.Graph.Inductive
import Data.Word
-- }}}

-- Debugging {{{
import Debug.Trace
echo_ :: Show α ⇒ α → α -- {{{
echo_ x = trace (show x) x
-- }}}
echoWithLabel_ :: Show α ⇒ String → α → α -- {{{
echoWithLabel_ label x = trace (label ++ " " ++ show x) x
-- }}}
-- }}}

-- Types {{{
type ContractionGraph = Gr Word Word
-- }}}

allContractionCosts :: MonadPlus m ⇒ Word → ContractionGraph → m Word -- {{{
allContractionCosts maximum_cost_so_far graph =
    case labNodes graph of
        [] → return maximum_cost_so_far
        first_node_choices → do
            (first_node,external_cost_1) ← msum . map return $ first_node_choices
            let (Just context1,graph_without_first) = match first_node graph
            case labNodes graph_without_first of
                [] → return maximum_cost_so_far
                second_node_choices → do
                    (second_node,external_cost_2) ← msum . map return $ second_node_choices
                    let (Just context2,graph_without_first_and_second) = match second_node graph_without_first
                        (incoming_adjacent_1,_,external_cost_1,outgoing_adjacent_1) = context1
                        (incoming_adjacent_2,_,external_cost_2,outgoing_adjacent_2) = context2
                        cost =
                            external_cost_1 *
                            external_cost_2 *
                            product (map fst incoming_adjacent_1) *
                            product (map fst incoming_adjacent_2) *
                            product (map fst outgoing_adjacent_1) *
                            product (map fst outgoing_adjacent_2)
                        incoming_adjacent_3 =
                            filter ((/= second_node) . snd) incoming_adjacent_1 ++
                            filter ((/= first_node) . snd) incoming_adjacent_2
                        external_cost_3 = external_cost_1 * external_cost_2
                        outgoing_adjacent_3 =
                            filter ((/= second_node) . snd) outgoing_adjacent_1 ++
                            filter ((/= first_node) . snd) outgoing_adjacent_2
                    allContractionCosts
                        (cost `max` maximum_cost_so_far)
                        ((incoming_adjacent_3,first_node,external_cost_3,outgoing_adjacent_3) & graph_without_first_and_second)
-- }}}
findMinimalContractionCost :: ContractionGraph → Word -- {{{
findMinimalContractionCost = minimum . allContractionCosts 0
-- }}}