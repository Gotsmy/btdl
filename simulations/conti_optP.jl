# -*- coding: utf-8 -*-
using JuMP
using LinearAlgebra
using Ipopt
using DelimitedFiles
using TickTock
using JLD2
import DataFrames
import CSV
import ForwardDiff
import Dierckx

include("/home/users/mgotsmy/julia/dFBA/custom/postprocessing_02.jl")

function fed_batch_dFBA(S,nFE,glu,atp,oxy,xxx,pro,aon,dil,vlb,vub,d)
    A_idx, B_idx, C_idx, D_idx = get_idx_lists(vlb,vub)
    nA = size(A_idx,1)
    nB = size(B_idx,1)
    nC = size(C_idx,1)
    nD = size(D_idx,1)
    println("nA = ",nA)
    println("nB = ",nB)
    println("nC = ",nC)
    println("nD = ",nD)
    
    # number of reactions
    nR = size(S,2)
    # number of metabolites
    nM = size(S,1)
    # number of tracked metabolites
    nN = 4
    
    #--------------------------
    # SIMULATION HYPERPARAMETERS
    
    w    = 1e-6 # weigth for flux sum minimization on the inner objective
    phi1 = 1e+0 # factor on outer objective
    phi2 = 1e-2 # factor on inner objective

    #--------------------------
    # JuMP MODEL SETUP
    m = Model(optimizer_with_attributes(Ipopt.Optimizer, 
            "warm_start_init_point" => "yes", 
            "print_level" => 5, 
            "linear_solver" => "ma27", 
            "max_iter" => Int(1e5),
            "tol" => 1e-8, 
            "acceptable_iter" => 50, 
            "acceptable_tol" => 1e-8))

    #--------------------------
    # VARIABLE SET UP
    @variables(m, begin
            N[1:nN, 1:nFE] # Metabolite concentrations: G, X, A, B
            q[1:nR, 1:nFE] # FBA fluxes [mmol/(g h)]

            # KKT Variables
            lambda_Sv[1:nM, 1:nFE]

            # A: both constrained
            alpha_ub_A[1:nA, 1:nFE]
            slack_ub_A[1:nA, 1:nFE]
            alpha_lb_A[1:nA, 1:nFE]
            slack_lb_A[1:nA, 1:nFE]

            # B: lb constrained, ub free
            alpha_lb_B[1:nB, 1:nFE]
            slack_lb_B[1:nB, 1:nFE]
            
            # C: lb free, ub constrained
            alpha_ub_C[1:nC, 1:nFE]
            slack_ub_C[1:nC, 1:nFE]
            
            # Process Variables
            Vdot
            Gf

    end)

    #--------------------------
    # GUESSING START VALUES
    
    println("Setting start values.")
    for i in 1:nFE
          for k in 1:nN
              set_start_value(N[k,i], 10)
          end
    end
    set_start_value(Vdot,1)
    println("Finished setting start values.")

    #--------------------------
    # SET UP OBJECTIVE FUNCTION AND CONSTRAINTS
    
    @NLobjective(m, Max, +
        Vdot*N[4,2]*phi1 -
	    sum(
	    sum(- slack_lb_A[mc,i] for mc in 1:nA) +
	    sum(- slack_ub_A[mc,i] for mc in 1:nA) +
	    sum(- slack_lb_B[mc,i] for mc in 1:nB) +
	    sum(- slack_ub_C[mc,i] for mc in 1:nC)
	    for i in 1:nFE)/nFE)
    
    @NLconstraints(m, begin
        # NL KKT constraints, aka complementary slackness
        NLc_slack_lb_A[mc=1:nA,i=1:nFE], slack_lb_A[mc,i]  == (q[A_idx[mc],i] -vlb[A_idx[mc]])*alpha_lb_A[mc,i]/nA*phi2
        NLc_slack_ub_A[mc=1:nA,i=1:nFE], slack_ub_A[mc,i]  == (q[A_idx[mc],i] -vub[A_idx[mc]])*alpha_ub_A[mc,i]/nA*phi2
        NLc_slack_lb_B[mc=1:nB,i=1:nFE], slack_lb_B[mc,i]  == (q[B_idx[mc],i] -vlb[B_idx[mc]])*alpha_lb_B[mc,i]/nB*phi2
        NLc_slack_ub_C[mc=1:nC,i=1:nFE], slack_ub_C[mc,i]  == (q[C_idx[mc],i] -vub[C_idx[mc]])*alpha_ub_C[mc,i]/nC*phi2

    end)

    #------------------------#
    # SET UP BOUNDS
    
    for mc in 1:nR
        for i in 1:nFE
            set_lower_bound(q[mc,i],vlb[mc])
            set_upper_bound(q[mc,i],vub[mc])
        end
    end
    
    for i in 1:nFE
        for mc in 1:nA
            set_upper_bound(alpha_lb_A[mc,i],0)
            set_lower_bound(alpha_ub_A[mc,i],0)
        end
        for mc in 1:nB
            set_upper_bound(alpha_lb_B[mc,i],0)
        end
        for mc in 1:nC
            set_lower_bound(alpha_ub_C[mc,i],0)
        end
    end
    
    #------------------------#
    # SET UP OTHER CONSTRAINTS
    
    @constraints(m, begin
            #------------------------#
            # DIFFERENTIAL EQUATIONS
            
            Adot1, -Vdot*N[3,1]               + q[aon,1]*N[2,1] == 0
            Adot2,  Vdot*N[3,1] - Vdot*N[3,2] + q[aon,2]*N[2,2] == 0
            Bdot1, -Vdot*N[4,1]               + q[pro,1]*N[2,1] == 0
            Bdot2,  Vdot*N[4,1] - Vdot*N[4,2] + q[pro,2]*N[2,2] == 0
            Xdot1, -Vdot*N[2,1]               + q[xxx,1]*N[2,1] == 0
            Xdot2,  Vdot*N[2,1] - Vdot*N[2,2] + q[xxx,2]*N[2,2] == 0
            Gdot1, -Vdot*N[1,1] + Vdot*Gf     + q[glu,1]*N[2,1] == 0
            Gdot2,  Vdot*N[1,1] - Vdot*N[1,2] + q[glu,2]*N[2,2] == 0
            
            #------------------------#
            # NON-NEGATIVITY CONSTRAINTS
            
            geqN[n=1:nN,i=1:nFE], N[n,i] >= 0
            geqVdot, Vdot >= 0
            geqGf, Gf >= 0
            leqGf, Gf <= 3000
            leqX[i=1:nFE], N[2,i] <= 16.7 # maximum biomass concentration
            leqG, N[1,2] == 0 # use up all glucose in the medium of reactor 2
            
            #------------------------#
            # PRODUCTION ENVELOPE
            
            c_qG[i=1:nFE],  q[glu,i]  +1.6832880759 +26.7371273713 * q[xxx,i] == 0 # µ  -> qG
            c_qA[i=1:nFE],  q[aon,i]  +5.0649596206 -36.6856368564 * q[xxx,i] >= 0 # µ  -> qA
            c_qD[i=1:nFE],  q[dil,i]  -1.2944612466 -10.3956639566 * q[xxx,i] == 0 # µ  -> qD
            c_qB[i=1:nFE],  q[pro,i] == q[dil,i]-q[aon,i]

            c_S[mc=1:nM,i=1:nFE], sum(S[mc,k] * q[k,i] for k in 1:nR) == 0

            #------------------------#
            # KKT CONSTRAINTS

            #Lagr_A[mc=1:nA,i=1:nFE],  d[A_idx[mc]] + w*q[A_idx[mc],i]  + alpha_lb_A[mc,i] + alpha_ub_A[mc,i] +  sum(S[k,A_idx[mc]]*lambda_Sv[k,i] for k in 1:nM) == 0
            #Lagr_B[mc=1:nB,i=1:nFE],  d[B_idx[mc]] + w*q[B_idx[mc],i]  + alpha_lb_B[mc,i]                    +  sum(S[k,B_idx[mc]]*lambda_Sv[k,i] for k in 1:nM) == 0
            #Lagr_C[mc=1:nC,i=1:nFE],  d[C_idx[mc]] + w*q[C_idx[mc],i]                     + alpha_ub_C[mc,i] +  sum(S[k,C_idx[mc]]*lambda_Sv[k,i] for k in 1:nM) == 0
            #Lagr_D[mc=1:nD,i=1:nFE],  d[D_idx[mc]] + w*q[D_idx[mc],i]                                        +  sum(S[k,D_idx[mc]]*lambda_Sv[k,i] for k in 1:nM) == 0
            # non-parsimonious    
            Lagr_A[mc=1:nA,i=1:nFE],  d[A_idx[mc]]  + alpha_lb_A[mc,i] + alpha_ub_A[mc,i] +  sum(S[k,A_idx[mc]]*lambda_Sv[k,i] for k in 1:nM) == 0
            Lagr_B[mc=1:nB,i=1:nFE],  d[B_idx[mc]]  + alpha_lb_B[mc,i]                    +  sum(S[k,B_idx[mc]]*lambda_Sv[k,i] for k in 1:nM) == 0
            Lagr_C[mc=1:nC,i=1:nFE],  d[C_idx[mc]]                     + alpha_ub_C[mc,i] +  sum(S[k,C_idx[mc]]*lambda_Sv[k,i] for k in 1:nM) == 0
            Lagr_D[mc=1:nD,i=1:nFE],  d[D_idx[mc]]                                        +  sum(S[k,D_idx[mc]]*lambda_Sv[k,i] for k in 1:nM) == 0
    end)

    #------------------------#
    # MODEL OPTIMIZATION
    
    println("Model Preprocessing Finished.")
    tick()
    solveNLP = JuMP.optimize!
    status = solveNLP(m)
    t = solve_time(m)
    println("Model Finished Optimization")

    # Read out model parameters
    Vdot_    = JuMP.value.(Vdot)
    Gf_ = JuMP.value(Gf)

    lambda_Sv_ = JuMP.value.(lambda_Sv[:,:])
    alpha_ub_A_ = JuMP.value.(alpha_ub_A[:,:])
    slack_ub_A_ = JuMP.value.(slack_ub_A[:,:])
    alpha_lb_A_ = JuMP.value.(alpha_lb_A[:,:])
    slack_lb_A_ = JuMP.value.(slack_lb_A[:,:])
    alpha_lb_B_ = JuMP.value.(alpha_lb_B[:,:])
    slack_lb_B_ = JuMP.value.(slack_lb_B[:,:])
    alpha_ub_C_ = JuMP.value.(alpha_ub_C[:,:])
    slack_ub_C_ = JuMP.value.(slack_ub_C[:,:])
    N_ = JuMP.value.(N[:,:])
    q_    = JuMP.value.(q[:,:])
    
    println("A ",N_[3,1]," ",N_[3,2])
    println("X ",N_[2,1]," ",N_[2,2])
    println("B ",N_[4,1]," ",N_[4,2])
    println("G ",N_[1,1]," ",N_[1,2])
    println("Objective ",Vdot_*N_[4,2])
    println("----------------")
    println("X0 = ",N_[2,1])
    println("Vdot = ",Vdot_)
    println("Gf = ",Gf_)
    println("g1, g2 = ",q_[glu,1]," , ",q_[glu,2])
    println("x1, x2 = ",q_[xxx,1]," , ",q_[xxx,2])
    println("d1, d2 = ",q_[dil,1]," , ",q_[dil,2])
    println("a1, a2 = ",q_[aon,1]," , ",q_[aon,2])
    println("b1, b2 = ",q_[pro,1]," , ",q_[pro,2])
    
    @JLD2.save dirname*"/variables.jld2" Vdot_ Gf_ N_ q_ lambda_Sv_ alpha_ub_A_ slack_ub_A_ alpha_lb_A_ slack_lb_A_ alpha_lb_B_ slack_lb_B_ alpha_ub_C_ slack_ub_C_ A_idx B_idx C_idx D_idx

    sum_slack = 0.0*Vector{Float64}(undef,nFE)
    for i in 1:nFE
        sum_slack[i] = sum(slack_lb_A_[j,i] for j in 1:nA; init=0) + 
                       sum(slack_ub_A_[j,i] for j in 1:nA; init=0) + 
                       sum(slack_lb_B_[j,i] for j in 1:nB; init=0) +
                       sum(slack_ub_C_[j,i] for j in 1:nC; init=0)
    end

return  N_, Vdot_, q_, sum_slack, m, t
end

function run_kkt_simulation()
    #------------------------#
    # LOAD METABOLIC MODEL
    S   = readdlm("/home/users/mgotsmy/julia/240426_extended_boku_model/analysis/btdl_pathway_S.csv")
    vlb = readdlm("/home/users/mgotsmy/julia/240426_extended_boku_model/analysis/btdl_pathway_LB.csv")[:,1]
    vub = readdlm("/home/users/mgotsmy/julia/240426_extended_boku_model/analysis/btdl_pathway_UB.csv")[:,1]
    println("nR = ",size(vlb,1)," = ",size(vub,1))
    
    # reaction indices
    glu = 49  # glucose uptake
    atp = 14  # atp maintenance
    xxx = 22  # biomass growth
    pro = 91  # butane23diol production
    oxy = 57  # oxygen uptake
    aon = 95  # acetoin exchange
    dil = 92  # diol production
    psk = 94  # butane23diol sink
    vlb[glu] = -6.7473 #mmol/(gX h)
    vub[glu] = -1.8143 #mmol/(gX h)
    nFE = 2 # this is now the number of reactors

    # KKT FBA objective function: optimize product
    nR = size(S,2)
    d  = 0.0*Vector{Float64}(undef,nR) 
    d[psk] = -1
    N_, Vdot_, q_, sum_slack, m_, t_ = fed_batch_dFBA(S,nFE,glu,atp,oxy,xxx,pro,aon,dil,vlb,vub,d)
    println("DONE")
          
    #-------------------------
    # SAVING ALL INTERESTING FILES
    
    summary = "#-------------------------\n" * 
              "Termination Status\n" * string(termination_status(m_)) * 
              "\n\nTime Elapsed\n" * string(t_) *
              "\n\nObjective Value\n" * string(JuMP.objective_value(m_)) * 
              "\n\nNormalized Complementary Slackness\n" * string(-(sum(-sum_slack))/nFE) * 
              "\n\nComplementary Slackness per Finite Element\n" * string([round(sum_slack[i],digits=5) for i in 1:nFE]) * 
              "\n\nFeed Rate\n" * string(round(Vdot_,digits=3)) *
              "\n\nGrowth Rates [h-1]\n" * string([round(i,digits=3) for i in q_[xxx,:]]) * 
              "\n\nFinal Product Amount [mmol/L]\n" * string(N_[4,end]) * 
              "\n\nFinal Biomass Amount [g/L]\n" * string(N_[2,end]) * 
              "\n#-------------------------"
    println(summary)
    
    #-------------------------
    # POST PROCESSING
    
    write(   dirname*"/summary.txt",summary)
    writedlm(dirname*"/q.csv",  q_, ',')
    writedlm(dirname*"/vlb.csv",vlb,',')
    writedlm(dirname*"/vub.csv",vub,',')
#     CSV.write(dirname*"/df.csv",df) 
end

# -------------------------
# SCRIPT START

println("Start Script")
scriptname = PROGRAM_FILE
dirname = scriptname[begin:end-3]
println("Creating results directory: ",dirname)
mkpath(dirname)

# -------------------------
# RUN MODEL

debug = true
if debug == true
    run_kkt_simulation()
else
    # sometimes errors happen during initialization, they usually desappear when rerunning the simulation
    # thus in debug = false mode, these errors are escapted.
    global worked = false
    global nTRY = 0
    while worked == false
        try
            run_kkt_simulation()
            global worked = true
        catch e
            if isa(e,ErrorException)
                global nTRY += 1
                println("Error During Initialization. Retrying ... (",nTRY,")")
            else
                rethrow(e)
            end
        end
    end
end

println("Script Ended")
